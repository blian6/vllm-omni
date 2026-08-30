# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone NPU platform mixin (no vllm-ascend dependency).

Provides the torch_npu-native implementations of the vLLM ``Platform``
interface that the standalone (no vllm-ascend) path needs, plus the
vLLM ``current_platform`` adoption logic used in worker subprocesses.

This mixin must stay free of any ``vllm_ascend`` import so it can be used
(and unit-tested) in environments without vllm-ascend installed.
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class StandaloneNPUPlatformMixin:
    """torch_npu-native ``Platform`` interface for standalone NPU runs.

    Pure diffusion stages (e.g. Qwen-Image) run with torch_npu + mindiesd
    attention and never need vllm-ascend; these methods cover the vLLM
    ``Platform`` interface entries that the standalone path must satisfy.
    """

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.npu.set_device(device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        device_props = torch.npu.get_device_properties(device_id)
        if not hasattr(device_props, "uuid") or device_props.uuid is None:
            raise RuntimeError(f"Device {device_id} does not have a valid UUID.")
        return device_props.uuid

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        props = torch.npu.get_device_properties(device_id)
        cube_core_num = getattr(props, "cube_core_num", None)
        if cube_core_num is not None and cube_core_num > 0:
            return int(cube_core_num)
        vector_core_num = getattr(props, "vector_core_num", None)
        if vector_core_num is not None and vector_core_num > 0:
            return int(vector_core_num)
        return 24  # safe default (24 Cube Cores)

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode()

    @classmethod
    def update_block_size_for_backend(cls, vllm_config) -> None:
        # Torch-NPU native path: keep vLLM's default block-size handling.
        return None

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.npu.manual_seed_all(seed)

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        torch.npu.reset_peak_memory_stats(device)
        return torch.npu.max_memory_allocated(device)

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    def adopt_as_vllm_platform(self) -> None:
        """Adopt this NPU platform as vLLM's ``current_platform``.

        vLLM 0.26 has no built-in NPU platform; without vllm-ascend its
        ``current_platform`` stays ``UnspecifiedPlatform``, and every
        ``from vllm.platforms import current_platform`` binds that object at
        import time (e.g. in ``vllm.utils.mem_utils``). We deliberately do
        NOT register a ``vllm.platform_plugins`` entry point (loading
        vllm_omni during vLLM's early plugin phase creates a circular import
        through ``vllm_omni.patch → vllm.config``). Instead, once the Omni
        platform is resolved, adopt it here so vLLM-side consumers (e.g.
        ``DeviceMemoryProfiler`` in diffusion worker subprocesses) see a real
        NPU platform. When vllm-ascend is installed vLLM already resolves its
        own NPUPlatform and this is a no-op.
        """
        from importlib.util import find_spec

        if find_spec("vllm_ascend") is not None:
            return
        try:
            import vllm.platforms as vllm_platforms

            if vllm_platforms.current_platform.is_unspecified():
                vllm_platforms.current_platform = self
                logger.debug(
                    "Adopted Omni platform as vLLM current_platform: %s",
                    type(self).__name__,
                )
                _rebind_vllm_platform_refs(self)
        except Exception:
            logger.debug("Failed to sync vLLM current_platform", exc_info=True)


def _rebind_vllm_platform_refs(platform) -> None:
    """Point already-imported vllm modules at the resolved Omni platform.

    vLLM 0.26 modules do ``from vllm.platforms import current_platform`` at
    import time, capturing whatever was resolved then. In worker subprocesses
    without vllm-ascend that is UnspecifiedPlatform; re-bind the captured
    references so vLLM-side helpers (DeviceMemoryProfiler, etc.) use the Omni
    NPU platform.
    """
    import sys

    module = sys.modules.get("vllm.utils.mem_utils")
    if module is None or not hasattr(module, "current_platform"):
        return
    bound = module.current_platform
    if bound is platform or getattr(bound, "device_type", "") != "":
        return
    module.current_platform = platform
    logger.debug("Re-bound %s.current_platform to %s", "vllm.utils.mem_utils", type(platform).__name__)
