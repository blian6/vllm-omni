# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vllm-ascend-enhanced NPU platform for AR/generation stages.

AR/generation stages (Qwen3-Omni, TTS, etc.) run on the vllm-ascend backend.
This module is only imported when the AR backend is selected (the platform
plugin resolves it from ``VLLM_OMNI_DISABLE_VLLM_ASCEND``, which defaults to
false), so the module-level ``vllm_ascend`` import is safe: it is only
reached when the vllm-ascend backend is actually required.
"""

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm_ascend.platform import NPUPlatform

from vllm_omni.platforms.npu.platform import NPUOmniPlatform, _vllm_ascend_available

logger = init_logger(__name__)


class ARNPUOmniPlatform(NPUOmniPlatform, NPUPlatform):
    """vllm-ascend implementation of the NPU platform.

    Inherits the vLLM ``Platform`` entries from vllm-ascend's ``NPUPlatform``
    and the shared interface from :class:`NPUOmniPlatform` (which only holds
    methods vllm-ascend does not define, so the MRO is conflict-free). The
    class body keeps the Ascend enhancements vllm-ascend does not provide:
    custom-op registration, ACL graph wrapper, ascend forward context, ascend
    config/logging, model patches, and the torch_npu total-memory override
    (vllm-ascend deliberately raises there, but ``startup_plan.py`` calls it
    unconditionally).
    """

    def __init__(self) -> None:
        if not _vllm_ascend_available():
            raise RuntimeError(
                "ARNPUOmniPlatform requires the vllm-ascend backend, but "
                "vllm-ascend is not installed. Pure diffusion stages do NOT "
                "need it; install vllm-ascend, or set "
                "VLLM_OMNI_DISABLE_VLLM_ASCEND=true in the stage env "
                "to use the standalone torch_npu backend (DiTNPUOmniPlatform)."
            )
        # Preserve the original application order: vllm-ascend global/model
        # patches first, then the 310P worker patches.
        from vllm_ascend.utils import adapt_patch

        from vllm_omni.platforms.npu._310p import apply_patches as apply_310p_patches
        from vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav import (
            apply_minicpmo_4_5_code2wav_patch,
        )
        from vllm_omni.platforms.npu.models.qwen3_tts_code2wav import (
            apply_qwen3_tts_code2wav_patch,
        )
        from vllm_omni.platforms.npu.models.qwen3_tts_tokenizer_v2 import (
            apply_qwen3_tts_tokenizer_v2_patch,
        )

        adapt_patch(is_global_patch=True)
        apply_minicpmo_4_5_code2wav_patch()
        apply_qwen3_tts_code2wav_patch()
        apply_qwen3_tts_tokenizer_v2_patch()
        apply_310p_patches()

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        super().set_device(device)

        # Register vllm_ascend custom ops (torch.ops._C_ascend.*).
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()

        # Ascend quantized weights are converted from ND to FRACTAL_NZ
        # after loading. Enable internal format so the NZ storage layout
        # is preserved for fused NPU kernels.
        torch.npu.config.allow_internal_format = True

    @classmethod
    def init_diffusion_worker_vllm_config(cls, vllm_config: Any) -> None:
        from vllm_ascend.ascend_config import init_ascend_config

        init_ascend_config(vllm_config)

    @classmethod
    def init_diffusion_model_runner_runtime(cls, vllm_config: Any, od_config: Any, device: torch.device) -> None:
        super().init_diffusion_model_runner_runtime(vllm_config, od_config, device)
        from vllm_ascend.ascend_forward_context import set_mc2_mask, set_mc2_tokens_capacity

        set_mc2_tokens_capacity(vllm_config, od_config.max_num_seqs, 1)
        set_mc2_mask(vllm_config, device)

    @classmethod
    def get_graph_wrapper_cls(cls) -> type:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        return ACLGraphWrapper

    @classmethod
    def set_forward_context(
        cls,
        attn_metadata,
        vllm_config,
        *,
        cudagraph_runtime_mode: CUDAGraphMode,
        batch_descriptor: BatchDescriptor,
    ):
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        return set_ascend_forward_context(
            attn_metadata,
            vllm_config,
            aclgraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        )

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Keep vllm-ascend's own config checks (parallel-config validation,
        # draft/decode context checks, incompatible-config fixes), then apply
        # the ascend config and logging setup.
        super().check_and_update_config(vllm_config)
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.logger import configure_ascend_file_logging, configure_ascend_logging

        init_ascend_config(vllm_config)
        configure_ascend_file_logging()
        configure_ascend_logging()

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # NOTE: vllm-ascend deliberately leaves this as NotImplementedError to
        # avoid initializing torch_npu too early, but vLLM's engine startup
        # (vllm/v1/worker/startup_plan.py) calls it unconditionally. Keep this
        # torch_npu implementation so the AR path satisfies the call.
        device_props = torch.npu.get_device_properties(device_id)
        return device_props.total_memory
