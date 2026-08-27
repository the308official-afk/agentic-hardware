from __future__ import annotations

from .base import SGLangHookTarget


HOOK_TARGETS: tuple[SGLangHookTarget, ...] = (
    SGLangHookTarget(
        module="sglang.srt.managers.cache_controller",
        class_name="HiCacheController",
        methods={
            "load": "hicache.load",
            "write": "hicache.write",
            "evict_device": "hicache.evict_device",
            "evict_host": "hicache.evict_host",
            "prefetch": "hicache.prefetch",
            "start_loading": "hicache.start_loading",
            "start_writing": "hicache.start_writing",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.hiradix_cache",
        class_name="HiRadixCache",
        methods={
            "match_prefix": "hiradix.match_prefix",
            "cache_finished_req": "hiradix.cache_finished_req",
            "cache_unfinished_req": "hiradix.cache_unfinished_req",
            "evict": "hiradix.evict",
            "load_back": "hiradix.load_back",
            "init_load_back": "hiradix.init_load_back",
            "ready_to_load_host_cache": "hiradix.ready_to_load_host_cache",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.radix_cache",
        class_name="RadixCache",
        methods={
            "match_prefix": "radix.match_prefix",
            "cache_finished_req": "radix.cache_finished_req",
            "cache_unfinished_req": "radix.cache_unfinished_req",
            "evict": "radix.evict",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="HostPoolGroup",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="MHATokenToKVPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="MLATokenToKVPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="NSATokenToKVPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="MambaPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.managers.scheduler",
        class_name="Scheduler",
        methods={
            "handle_generate_request": "scheduler.handle_generate_request",
            "_add_request_to_queue": "scheduler.add_request_to_queue",
            "_prefetch_kvcache": "scheduler.prefetch_kvcache",
            "_run_batch_prebuilt": "scheduler.run_batch_prebuilt",
            "process_batch_result": "scheduler.process_batch_result",
            "process_batch_result_prefill": "scheduler.process_batch_result_prefill",
            "process_batch_result_decode": "scheduler.process_batch_result_decode",
            "run_batch": "scheduler.run_batch",
            "process_input_requests": "scheduler.process_input_requests",
            "get_next_batch_to_run": "scheduler.get_next_batch_to_run",
            "get_new_batch_prefill": "scheduler.get_new_batch_prefill",
            "event_loop_overlap": "scheduler.event_loop_overlap",
            "event_loop_normal": "scheduler.event_loop_normal",
        },
        scheduler_required=True,
    ),
    SGLangHookTarget(
        module="sglang.srt.managers.tp_worker",
        class_name="TpModelWorker",
        methods={
            "forward_batch_generation": "worker.forward_batch_generation",
            "forward_batch_split_prefill": "worker.forward_batch_split_prefill",
            "_forward_batch_generation_dllm": "worker.forward_batch_generation_dllm",
            "forward_batch_embedding": "worker.forward_batch_embedding",
        },
        scheduler_required=True,
    ),
)


RAW_EVENT_MAP: dict[str, str] = {
    "hicache.write.end": "KV_WRITE_HOST",
    "hicache.evict_device.end": "KV_EVICT_GPU",
    "hicache.evict_host.end": "KV_EVICT_HOST",
    "hicache.load.end": "KV_LOAD_GPU",
    "hostpool.load_to_device_per_layer.end": "KV_LOAD_GPU",
    "hostpool.backup_from_device_all_layer.end": "KV_WRITE_HOST",
    "hiradix.init_load_back.end": "KV_LOAD_GPU",
    "hiradix.load_back.end": "KV_LOAD_GPU",
    "hiradix.match_prefix.end": "KV_MATCH_PREFIX",
}
