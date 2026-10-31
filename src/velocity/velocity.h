/* Velocity: zllm's native Direct3D 12 host (src/velocity/host.cpp).
 *
 * A thin C interface over D3D12 with the Agility SDK and DirectStorage:
 * bindless raw buffers, one root signature of root constants, DXIL
 * pipelines, ticketed compute submission with timestamps, and DirectStorage
 * range reads from files straight into buffers. The Zig host (host.zig)
 * executes zllm's command stream on it. Windows only; compiled into zllm.
 */
#ifndef ZLLM_VELOCITY_H
#define ZLLM_VELOCITY_H

#include <stddef.h>
#include <stdint.h>

#define VEL_API

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vel_context vel_context;
typedef struct vel_file vel_file;
typedef struct vel_buffer vel_buffer;
typedef struct vel_pipeline vel_pipeline;
typedef int32_t vel_result;

typedef struct vel_options {
    uint32_t adapter_index;
    uint32_t enable_debug;
    uint32_t require_linalg;
    uint32_t staging_bytes;
    uint32_t queue_capacity;
} vel_options;

typedef struct vel_caps {
    uint32_t abi_version;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t shader_model;
    uint32_t linalg_tier;
    uint32_t wave16_fp16_fp32_flags;
    uint32_t vector_fp16_fp32_flags;
    uint32_t experimental_features_enabled;
    uint64_t dedicated_video_bytes;
    uint64_t local_budget_bytes;
    uint64_t local_usage_bytes;
    char adapter_name[128];
} vel_caps;

typedef struct vel_range {
    uint64_t file_offset;
    uint64_t buffer_offset;
    uint32_t bytes;
    uint32_t reserved;
} vel_range;

enum { VEL_RESOURCE_DEVICE = 0, VEL_RESOURCE_HOST = 1 };

typedef struct vel_buffer_info {
    uint32_t srv_index;
    uint32_t uav_index;
    void *mapped;
    uint64_t gpu_address;
    uint64_t allocated_bytes;
} vel_buffer_info;

/* All methods require serialized access to each context.
 * Results use HRESULT semantics. Negative values indicate failure.
 * Buffers remain in COMMON between complete operations.
 * A failed or timed-out destroy/release retains the handle for a later retry.
 * Use separate buffers for cache slots with independent resource lifetimes.
 */
VEL_API vel_result vel_create(const vel_options *options, vel_context **out_context);
VEL_API vel_result vel_get_caps(vel_context *context, vel_caps *out_caps);
VEL_API vel_result vel_open_file(vel_context *context, const char *utf8_path, vel_file **out_file);
VEL_API vel_result vel_file_size(vel_file *file, uint64_t *out_bytes);
VEL_API vel_result vel_create_buffer(vel_context *context, uint64_t bytes, vel_buffer **out_buffer);

/* This method enqueues raw byte transfers. It does not decode Gyro weights.
 * Ranges must not overlap in the destination. Each range must fit staging_bytes.
 * ERROR_BUSY means that this buffer still has an earlier storage request.
 * The batch waits for this buffer's last compute fence before any GPU work.
 */
VEL_API vel_result vel_read_ranges(vel_context *context, vel_file *file, vel_buffer *buffer,
    const vel_range *ranges, uint32_t count, uint64_t *out_storage_ticket);
VEL_API vel_result vel_poll(vel_context *context, uint64_t *storage_completed, uint64_t *compute_completed);
VEL_API vel_result vel_wait_storage(vel_context *context, uint64_t ticket, uint32_t timeout_ms);
VEL_API vel_result vel_readback(vel_context *context, vel_buffer *buffer, uint64_t offset,
    uint32_t bytes, void *destination, uint32_t timeout_ms);
/* Bindless compute API (requires Shader Model 6.6).
 * Every resource receives a raw SRV and a raw UAV in one shader-visible heap; shaders index
 * ResourceDescriptorHeap with the returned indices. Pipelines share one root signature of
 * 60 root constants at b0. VEL_RESOURCE_DEVICE allocates local video memory. VEL_RESOURCE_HOST
 * allocates persistently mapped write-back system memory that shaders can read and write.
 * Buffers rely on implicit state promotion: within one submission, read a buffer only through
 * its SRV or only through its UAV. vel_barrier inserts a global UAV barrier.
 * A buffer may receive DirectStorage writes into ranges that no in-flight compute reads.
 */
VEL_API vel_result vel_create_resource(vel_context *context, uint64_t bytes, uint32_t kind,
    vel_buffer **out_buffer, vel_buffer_info *out_info);
VEL_API vel_result vel_create_pipeline(vel_context *context, const void *dxil, size_t dxil_bytes,
    vel_pipeline **out_pipeline);
/* Compiles HLSL (entry point "main") for `target` (e.g. "cs_6_6") with
 * velocity\dxcompiler.dll beside the executable, then creates its pipeline.
 * On a compile error, vel_last_error holds the compiler's messages. */
VEL_API vel_result vel_compile_pipeline(vel_context *context, const char *source, size_t bytes,
    const char *target, vel_pipeline **out_pipeline);
/* Copies the last error message (NUL-terminated); returns its length. */
VEL_API size_t vel_last_error(vel_context *context, char *out, size_t capacity);
/* Removes the device, as a driver fault would (lost-device tests). */
VEL_API vel_result vel_remove_device(vel_context *context);
VEL_API vel_result vel_release_pipeline(vel_context *context, vel_pipeline *pipeline, uint32_t timeout_ms);
VEL_API vel_result vel_begin(vel_context *context, uint32_t timeout_ms);
VEL_API vel_result vel_dispatch(vel_context *context, vel_pipeline *pipeline, const uint32_t *constants,
    uint32_t constant_count, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
VEL_API vel_result vel_barrier(vel_context *context);
VEL_API vel_result vel_timestamp(vel_context *context, uint32_t *out_index);
VEL_API vel_result vel_submit(vel_context *context, uint64_t *out_compute_ticket);
VEL_API vel_result vel_wait_compute(vel_context *context, uint64_t ticket, uint32_t timeout_ms);
VEL_API vel_result vel_read_timestamps(vel_context *context, uint32_t count, uint64_t *out_ticks,
    uint64_t *out_frequency);
/* Makes later compute submissions wait on the GPU for a storage ticket. */
VEL_API vel_result vel_queue_wait_storage(vel_context *context, uint64_t storage_ticket);
/* Arena variant of vel_read_ranges: the caller names the compute ticket that must finish before
 * DirectStorage writes, normally the last use of the evicted slots. No per-buffer busy check. */
VEL_API vel_result vel_read_ranges_after(vel_context *context, vel_file *file, vel_buffer *buffer,
    const vel_range *ranges, uint32_t count, uint64_t compute_ticket, uint64_t *out_storage_ticket);
VEL_API vel_result vel_release_buffer(vel_context *context, vel_buffer *buffer, uint32_t timeout_ms);
VEL_API vel_result vel_release_file(vel_context *context, vel_file *file, uint32_t timeout_ms);
VEL_API vel_result vel_destroy(vel_context *context, uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif
#endif
