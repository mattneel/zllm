// Agility's native half (see agility.h). Compiled by zig c++ for
// x86_64-windows-gnu against the pinned Agility SDK and DirectStorage headers.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dstorage.h>
#include <dxcapi.h>
#include "mingw_uuid.h"
#include "agility.h"

// The Agility SDK release staged in .\\D3D12\\ (scripts/fixtures.sh agility_ver).
static constexpr UINT agility_version = 722;

#ifndef E_ILLEGAL_METHOD_CALL // not in MinGW's winerror.h
#define E_ILLEGAL_METHOD_CALL _HRESULT_TYPEDEF_(0x8000000EL)
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

static_assert(sizeof(vel_options) == 20);
static_assert(sizeof(vel_caps) == 184);
static_assert(sizeof(vel_range) == 24);
static_assert(offsetof(vel_caps, adapter_name) == 56);
static_assert(sizeof(vel_buffer_info) == 32);
static_assert(offsetof(vel_buffer_info, mapped) == 8);

template<class T> struct Com {
    T *p = nullptr;
    Com() = default;
    ~Com() { if (p) p->Release(); }
    Com(const Com&) = delete;
    Com& operator=(const Com&) = delete;
    Com(Com&& other) noexcept : p(std::exchange(other.p, nullptr)) {}
    Com& operator=(Com&& other) noexcept {
        if (this != &other) { if (p) p->Release(); p = std::exchange(other.p, nullptr); }
        return *this;
    }
    T *operator->() const { return p; }
    T **put() { return &p; }
};

struct vel_file {
    vel_context *owner;
    Com<IDStorageFile> file;
    uint64_t bytes = 0;
    uint64_t last_storage = 0;
};
struct vel_buffer {
    vel_context *owner;
    Com<ID3D12Resource> resource;
    uint64_t bytes = 0;
    uint64_t last_storage = 0;
    uint64_t last_compute = 0;
    uint32_t srv_index = UINT32_MAX;
    uint32_t uav_index = UINT32_MAX;
    void *mapped = nullptr;
};
struct vel_pipeline {
    vel_context *owner;
    Com<ID3D12PipelineState> pso;
    uint32_t constant_dwords = 0;
};
struct Frame {
    Com<ID3D12CommandAllocator> allocator;
    Com<ID3D12GraphicsCommandList> list;
    uint64_t ticket = 0;
};
struct Batch {
    uint64_t ticket;
    Com<IDStorageStatusArray> status;
};
struct vel_context {
    Com<IDXGIFactory6> factory;
    Com<IDXGIAdapter1> adapter;
    Com<ID3D12Device> device;
    Com<ID3D12CommandQueue> compute;
    Com<ID3D12CommandAllocator> allocator;
    Com<ID3D12GraphicsCommandList> list;
    Com<ID3D12Fence> storage_fence;
    Com<ID3D12Fence> compute_fence;
    Com<IDStorageFactory> storage_factory;
    Com<IDStorageQueue3> storage;
    HANDLE event = nullptr;
    vel_caps caps{};
    uint32_t staging_bytes = 32 * 1024 * 1024;
    uint32_t queue_capacity = 1024;
    uint64_t next_storage = 0;
    uint64_t next_compute = 0;
    HRESULT storage_error = S_OK;
    std::vector<Batch> batches;
    std::vector<vel_file*> files;
    std::vector<vel_buffer*> buffers;
    std::vector<Com<IUnknown>> compute_hold;
    // Bindless compute state: one shader-visible heap, one root signature of root constants.
    Com<ID3D12DescriptorHeap> heap;
    Com<ID3D12RootSignature> bindless_root;
    UINT descriptor_size = 0;
    uint32_t heap_capacity = 0;
    uint32_t heap_next = 0;
    std::vector<uint32_t> heap_free;
    Frame frames[3];
    int recording = -1;
    uint32_t next_frame = 0;
    uint64_t last_bindless_compute = 0;
    std::vector<vel_pipeline*> pipelines;
    Com<ID3D12QueryHeap> timestamps;
    Com<ID3D12Resource> timestamp_readback;
    uint32_t timestamp_count = 0;
    uint64_t timestamp_frequency = 0;
    // The HLSL compiler (agility\\dxcompiler.dll next to the executable), loaded on first use.
    HMODULE dxc_module = nullptr;
    Com<IDxcUtils> dxc_utils;
    Com<IDxcCompiler3> dxc;
    Com<IDxcIncludeHandler> dxc_includes;
    std::wstring dxc_include_path;
    std::string last_error;
    ~vel_context() {
        for (auto *p : pipelines) delete p;
        for (auto *b : buffers) delete b;
        for (auto *f : files) delete f;
        dxc_includes = {};
        dxc = {};
        dxc_utils = {};
        if (dxc_module) FreeLibrary(dxc_module);
        if (event) CloseHandle(event);
    }
};

#define VEL_TRY(expr) do { HRESULT vel_hr = (expr); if (FAILED(vel_hr)) return vel_hr; } while (0)
#define VEL_CATCH catch (const std::bad_alloc&) { return E_OUTOFMEMORY; } catch (...) { return E_FAIL; }

static HRESULT wait_fence(vel_context *c, ID3D12Fence *fence, uint64_t ticket, uint32_t timeout) {
    uint64_t complete = fence->GetCompletedValue();
    if (complete == UINT64_MAX) return DXGI_ERROR_DEVICE_REMOVED;
    if (complete >= ticket) return S_OK;
    VEL_TRY(fence->SetEventOnCompletion(ticket, c->event));
    const ULONGLONG start = GetTickCount64();
    for (;;) {
        ULONGLONG elapsed = GetTickCount64() - start;
        DWORD remaining = timeout == INFINITE ? INFINITE :
            (elapsed >= timeout ? 0 : timeout - static_cast<DWORD>(elapsed));
        DWORD result = WaitForSingleObject(c->event, remaining);
        complete = fence->GetCompletedValue();
        if (complete == UINT64_MAX) return DXGI_ERROR_DEVICE_REMOVED;
        if (complete >= ticket) return S_OK;
        if (result == WAIT_TIMEOUT) return HRESULT_FROM_WIN32(WAIT_TIMEOUT);
        if (result != WAIT_OBJECT_0) return HRESULT_FROM_WIN32(GetLastError());
        // A previous timed-out registration can signal this shared event.
    }
}

// As wait_fence, for teardown: a removed device runs nothing more, so its
// work counts as finished and its objects can be released.
static HRESULT settle(vel_context *c, ID3D12Fence *fence, uint64_t ticket, uint32_t timeout) {
    HRESULT hr = wait_fence(c, fence, ticket, timeout);
    return hr == DXGI_ERROR_DEVICE_REMOVED ? S_OK : hr;
}

static HRESULT retire_batches(vel_context *c, uint64_t complete) {
    size_t count = 0;
    bool status_pending = false;
    for (auto &batch : c->batches) {
        if (batch.ticket > complete) break;
        HRESULT hr = batch.status->GetHResult(0);
        if (hr == E_PENDING) { status_pending = true; break; }
        if (FAILED(hr) && SUCCEEDED(c->storage_error)) c->storage_error = hr;
        ++count;
    }
    c->batches.erase(c->batches.begin(), c->batches.begin() + count);
    if (FAILED(c->storage_error)) return c->storage_error;
    return status_pending ? E_PENDING : S_OK;
}

static HRESULT make_resource(vel_context *c, uint64_t bytes, D3D12_HEAP_TYPE type,
    D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags, Com<ID3D12Resource> &out) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return c->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state,
        nullptr, IID_PPV_ARGS(out.put()));
}

static void transition(vel_context *c, ID3D12Resource *resource,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    c->list->ResourceBarrier(1, &barrier);
}

static D3D12_CPU_DESCRIPTOR_HANDLE cpu_heap_start(ID3D12DescriptorHeap *heap) {
#if defined(_MSC_VER) || !defined(_WIN32)
    return heap->GetCPUDescriptorHandleForHeapStart();
#else
    D3D12_CPU_DESCRIPTOR_HANDLE result{};
    heap->GetCPUDescriptorHandleForHeapStart(&result);
    return result;
#endif
}

static D3D12_GPU_DESCRIPTOR_HANDLE gpu_heap_start(ID3D12DescriptorHeap *heap) {
#if defined(_MSC_VER) || !defined(_WIN32)
    return heap->GetGPUDescriptorHandleForHeapStart();
#else
    D3D12_GPU_DESCRIPTOR_HANDLE result{};
    heap->GetGPUDescriptorHandleForHeapStart(&result);
    return result;
#endif
}

static void hold(vel_context *c, IUnknown *object) {
    c->compute_hold.emplace_back();
    object->AddRef();
    c->compute_hold.back().p = object;
}

static HRESULT begin_compute(vel_context *c, uint32_t timeout) {
    VEL_TRY(wait_fence(c, c->compute_fence.p, c->next_compute, timeout));
    c->compute_hold.clear();
    VEL_TRY(c->allocator->Reset());
    return c->list->Reset(c->allocator.p, nullptr);
}

static HRESULT submit_compute(vel_context *c, uint64_t *ticket) {
    if (c->next_compute == UINT64_MAX - 1) return E_UNEXPECTED;
    VEL_TRY(c->list->Close());
    ID3D12CommandList *lists[] = {c->list.p};
    c->compute->ExecuteCommandLists(1, lists);
    *ticket = ++c->next_compute;
    return c->compute->Signal(c->compute_fence.p, *ticket);
}

// Bindless compute requires SM 6.6 ResourceDescriptorHeap indexing. Older adapters keep the
// validation entry points and report DXGI_ERROR_UNSUPPORTED from the compute API.
static constexpr uint32_t bindless_constant_dwords = 60;
static HRESULT init_bindless(vel_context *c) {
    if (c->caps.shader_model < D3D_SHADER_MODEL_6_6) return S_OK;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 1u << 16;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    VEL_TRY(c->device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(c->heap.put())));
    c->heap_capacity = heap_desc.NumDescriptors;
    c->descriptor_size = c->device->GetDescriptorHandleIncrementSize(heap_desc.Type);
    D3D12_ROOT_PARAMETER1 param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, bindless_constant_dwords};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    root_desc.Desc_1_1.NumParameters = 1;
    root_desc.Desc_1_1.pParameters = &param;
    root_desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
    Com<ID3DBlob> serialized, errors;
    VEL_TRY(D3D12SerializeVersionedRootSignature(&root_desc, serialized.put(), errors.put()));
    VEL_TRY(c->device->CreateRootSignature(0, serialized->GetBufferPointer(),
        serialized->GetBufferSize(), IID_PPV_ARGS(c->bindless_root.put())));
    for (auto &frame : c->frames) {
        VEL_TRY(c->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(frame.allocator.put())));
        VEL_TRY(c->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, frame.allocator.p, nullptr,
            IID_PPV_ARGS(frame.list.put())));
        VEL_TRY(frame.list->Close());
    }
    D3D12_QUERY_HEAP_DESC query{};
    query.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    query.Count = 4096;
    VEL_TRY(c->device->CreateQueryHeap(&query, IID_PPV_ARGS(c->timestamps.put())));
    VEL_TRY(make_resource(c, uint64_t(query.Count) * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_FLAG_NONE, c->timestamp_readback));
    return c->compute->GetTimestampFrequency(&c->timestamp_frequency);
}

extern "C" vel_result vel_create(const vel_options *options, vel_context **out) {
    if (!out) return E_INVALIDARG;
    *out = nullptr;
    try {
        vel_options opts{};
        if (options) opts = *options;
        auto c = std::make_unique<vel_context>();
        if (opts.staging_bytes) c->staging_bytes = opts.staging_bytes;
        if (opts.queue_capacity) c->queue_capacity = opts.queue_capacity;
        if (c->queue_capacity < 128 || c->queue_capacity > DSTORAGE_MAX_QUEUE_CAPACITY ||
            (c->queue_capacity & (c->queue_capacity - 1)) != 0) return E_INVALIDARG;
        if (c->staging_bytes < 4096) return E_INVALIDARG;
        // Agility's device comes from the Agility SDK runtime (.\D3D12\D3D12Core.dll beside the
        // executable) through a device factory, with experimental shader models (LinAlg) enabled
        // on that factory only: the rest of the process (Dawn) keeps the system runtime.
        // MinGW's d3d12 import library lacks D3D12GetInterface.
        auto get_interface = reinterpret_cast<PFN_D3D12_GET_INTERFACE>(
            reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"d3d12.dll"), "D3D12GetInterface")));
        if (!get_interface) return HRESULT_FROM_WIN32(ERROR_PROC_NOT_FOUND);
        Com<ID3D12SDKConfiguration1> sdk;
        VEL_TRY(get_interface(CLSID_D3D12SDKConfiguration, IID_PPV_ARGS(sdk.put())));
        Com<ID3D12DeviceFactory> device_factory;
        VEL_TRY(sdk->CreateDeviceFactory(agility_version, ".\\D3D12\\", IID_PPV_ARGS(device_factory.put())));
        if (opts.enable_debug) {
            Com<ID3D12Debug> debug;
            VEL_TRY(device_factory->GetConfigurationInterface(CLSID_D3D12Debug, IID_PPV_ARGS(debug.put())));
            debug->EnableDebugLayer();
        }
        const UUID features[] = {D3D12ExperimentalShaderModels};
        HRESULT experimental = device_factory->EnableExperimentalFeatures(1, features, nullptr, nullptr);
        if (FAILED(experimental) && opts.require_linalg) return experimental;
        c->caps.experimental_features_enabled = SUCCEEDED(experimental);
        VEL_TRY(CreateDXGIFactory2(opts.enable_debug ? DXGI_CREATE_FACTORY_DEBUG : 0,
            IID_PPV_ARGS(c->factory.put())));
        VEL_TRY(c->factory->EnumAdapterByGpuPreference(opts.adapter_index,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(c->adapter.put())));
        VEL_TRY(device_factory->CreateDevice(c->adapter.p, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(c->device.put())));
        DXGI_ADAPTER_DESC1 desc{};
        VEL_TRY(c->adapter->GetDesc1(&desc));
        c->caps.abi_version = 1;
        c->caps.vendor_id = desc.VendorId;
        c->caps.device_id = desc.DeviceId;
        c->caps.dedicated_video_bytes = desc.DedicatedVideoMemory;
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, c->caps.adapter_name,
            sizeof(c->caps.adapter_name), nullptr, nullptr);
        c->caps.adapter_name[sizeof(c->caps.adapter_name) - 1] = 0;
        Com<IDXGIAdapter3> adapter3;
        if (SUCCEEDED(c->adapter->QueryInterface(IID_PPV_ARGS(adapter3.put())))) {
            DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
            if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory))) {
                c->caps.local_budget_bytes = memory.Budget;
                c->caps.local_usage_bytes = memory.CurrentUsage;
            }
        }
        // The runtime rejects shader models it does not know, so query from the newest downward.
        const D3D_SHADER_MODEL models[] = {D3D_SHADER_MODEL_6_10, D3D_SHADER_MODEL_6_9, D3D_SHADER_MODEL_6_8,
            D3D_SHADER_MODEL_6_7, D3D_SHADER_MODEL_6_6, D3D_SHADER_MODEL_6_5, D3D_SHADER_MODEL_6_0};
        for (D3D_SHADER_MODEL model : models) {
            D3D12_FEATURE_DATA_SHADER_MODEL shader{model};
            if (SUCCEEDED(c->device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shader, sizeof(shader)))) {
                c->caps.shader_model = shader.HighestShaderModel;
                break;
            }
        }
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT tier{};
        if (SUCCEEDED(c->device->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, &tier, sizeof(tier))))
            c->caps.linalg_tier = tier.LinearAlgebraTier;
        D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT op{};
        op.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_WAVE_MATRIX_MULTIPLY;
        op.WaveMatrixMultiply.Inputs = {32, D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16,
            D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16, D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32};
        op.WaveMatrixMultiply.Shape = {16, 16, 16};
        if (SUCCEEDED(c->device->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT, &op, sizeof(op))))
            c->caps.wave16_fp16_fp32_flags = op.WaveMatrixMultiply.SupportFlags;
        op = {};
        op.OperationType = D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY;
        op.ThreadVectorMatrixMultiply.VectorInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        op.ThreadVectorMatrixMultiply.MatrixInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16;
        op.ThreadVectorMatrixMultiply.BiasInputType = D3D12_LINEAR_ALGEBRA_DATATYPE_NONE;
        op.ThreadVectorMatrixMultiply.VectorResultType = D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32;
        if (SUCCEEDED(c->device->CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT, &op, sizeof(op))))
            c->caps.vector_fp16_fp32_flags = op.ThreadVectorMatrixMultiply.SupportFlags;
        if (opts.require_linalg && (c->caps.shader_model < D3D_SHADER_MODEL_6_10 ||
            c->caps.linalg_tier < D3D12_LINEAR_ALGEBRA_TIER_1_0)) return DXGI_ERROR_UNSUPPORTED;
        D3D12_COMMAND_QUEUE_DESC qdesc{};
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        VEL_TRY(c->device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(c->compute.put())));
        VEL_TRY(c->device->CreateCommandAllocator(qdesc.Type, IID_PPV_ARGS(c->allocator.put())));
        VEL_TRY(c->device->CreateCommandList(0, qdesc.Type, c->allocator.p, nullptr, IID_PPV_ARGS(c->list.put())));
        VEL_TRY(c->list->Close());
        VEL_TRY(c->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(c->storage_fence.put())));
        VEL_TRY(c->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(c->compute_fence.put())));
        c->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!c->event) return HRESULT_FROM_WIN32(GetLastError());
        VEL_TRY(DStorageGetFactory(IID_PPV_ARGS(c->storage_factory.put())));
        VEL_TRY(c->storage_factory->SetStagingBufferSize(c->staging_bytes));
        DSTORAGE_QUEUE_DESC storage_desc{};
        storage_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        storage_desc.Capacity = static_cast<UINT16>(c->queue_capacity);
        storage_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
        storage_desc.Name = "zllm-reads";
        storage_desc.Device = c->device.p;
        VEL_TRY(c->storage_factory->CreateQueue(&storage_desc, IID_PPV_ARGS(c->storage.put())));
        VEL_TRY(init_bindless(c.get()));
        *out = c.release();
        return S_OK;
    } VEL_CATCH
}

extern "C" vel_result vel_get_caps(vel_context *c, vel_caps *out) {
    if (!c || !out) return E_INVALIDARG;
    Com<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(c->adapter->QueryInterface(IID_PPV_ARGS(adapter3.put())))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory))) {
            c->caps.local_budget_bytes = memory.Budget;
            c->caps.local_usage_bytes = memory.CurrentUsage;
        }
    }
    *out = c->caps;
    return S_OK;
}

extern "C" vel_result vel_open_file(vel_context *c, const char *utf8, vel_file **out) {
    if (!c || !utf8 || !out) return E_INVALIDARG;
    *out = nullptr;
    try {
        int chars = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, nullptr, 0);
        if (!chars) return HRESULT_FROM_WIN32(GetLastError());
        std::vector<wchar_t> path(static_cast<size_t>(chars));
        if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, path.data(), chars))
            return HRESULT_FROM_WIN32(GetLastError());
        auto f = std::make_unique<vel_file>();
        f->owner = c;
        VEL_TRY(c->storage_factory->OpenFile(path.data(), IID_PPV_ARGS(f->file.put())));
        BY_HANDLE_FILE_INFORMATION info{};
        VEL_TRY(f->file->GetFileInformation(&info));
        f->bytes = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
        c->files.push_back(f.get());
        *out = f.release();
        return S_OK;
    } VEL_CATCH
}

extern "C" vel_result vel_file_size(vel_file *f, uint64_t *out) {
    if (!f || !out) return E_INVALIDARG;
    *out = f->bytes;
    return S_OK;
}

extern "C" vel_result vel_create_buffer(vel_context *c, uint64_t bytes, vel_buffer **out) {
    if (!c || !out || !bytes || bytes > UINT64_MAX - 15) return E_INVALIDARG;
    *out = nullptr;
    try {
        auto b = std::make_unique<vel_buffer>();
        b->owner = c;
        b->bytes = bytes;
        VEL_TRY(make_resource(c, (bytes + 15) & ~uint64_t(15), D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_NONE, b->resource));
        c->buffers.push_back(b.get());
        *out = b.release();
        return S_OK;
    } VEL_CATCH
}

extern "C" vel_result vel_poll(vel_context *c, uint64_t *storage_complete, uint64_t *compute_complete) {
    if (!c || !storage_complete || !compute_complete) return E_INVALIDARG;
    *storage_complete = c->storage_fence->GetCompletedValue();
    *compute_complete = c->compute_fence->GetCompletedValue();
    if (*storage_complete == UINT64_MAX || *compute_complete == UINT64_MAX) return DXGI_ERROR_DEVICE_REMOVED;
    return retire_batches(c, *storage_complete);
}

extern "C" vel_result vel_wait_storage(vel_context *c, uint64_t ticket, uint32_t timeout) {
    if (!c || ticket > c->next_storage) return E_INVALIDARG;
    VEL_TRY(wait_fence(c, c->storage_fence.p, ticket, timeout));
    return retire_batches(c, c->storage_fence->GetCompletedValue());
}

extern "C" vel_result vel_read_ranges(vel_context *c, vel_file *f, vel_buffer *b,
    const vel_range *ranges, uint32_t count, uint64_t *ticket) {
    if (!c || !f || !b || !ranges || !ticket || f->owner != c || b->owner != c ||
        !count || count > c->queue_capacity / 2 - 2) return E_INVALIDARG;
    *ticket = 0;
    try {
        uint64_t io_complete, compute_complete;
        VEL_TRY(vel_poll(c, &io_complete, &compute_complete));
        if (b->last_storage > io_complete) return HRESULT_FROM_WIN32(ERROR_BUSY);
        if (c->next_storage == UINT64_MAX - 1) return E_UNEXPECTED;
        std::vector<DSTORAGE_REQUEST> requests(count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto &r = ranges[i];
            if (!r.bytes || r.reserved || r.bytes > c->staging_bytes ||
                r.file_offset > f->bytes || r.bytes > f->bytes - r.file_offset ||
                r.buffer_offset > b->bytes || r.bytes > b->bytes - r.buffer_offset) return E_INVALIDARG;
            for (uint32_t j = 0; j < i; ++j) {
                const auto &s = ranges[j];
                if (r.buffer_offset < s.buffer_offset + s.bytes &&
                    s.buffer_offset < r.buffer_offset + r.bytes) return E_INVALIDARG;
            }
            auto &request = requests[i];
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            request.Source.File = {f->file.p, r.file_offset, r.bytes};
            request.Destination.Buffer = {b->resource.p, r.buffer_offset, r.bytes};
            request.Name = "zllm-range";
        }
        Batch batch{};
        batch.ticket = c->next_storage + 1;
        VEL_TRY(c->storage_factory->CreateStatusArray(1, "zllm-range-status", IID_PPV_ARGS(batch.status.put())));
        c->batches.push_back(std::move(batch));
        c->storage->EnqueueRequests(requests.data(), count, c->compute_fence.p,
            b->last_compute, DSTORAGE_ENQUEUE_REQUEST_FLAG_FENCE_WAIT_BEFORE_GPU_WORK);
        c->storage->EnqueueStatus(c->batches.back().status.p, 0);
        c->storage->EnqueueSignal(c->storage_fence.p, ++c->next_storage);
        c->storage->Submit();
        b->last_storage = f->last_storage = *ticket = c->next_storage;
        return S_OK;
    } VEL_CATCH
}

extern "C" vel_result vel_readback(vel_context *c, vel_buffer *b, uint64_t offset,
    uint32_t bytes, void *destination, uint32_t timeout) {
    if (!c || !b || b->owner != c || !destination || !bytes || offset > b->bytes ||
        bytes > b->bytes - offset) return E_INVALIDARG;
    try {
        VEL_TRY(vel_wait_storage(c, b->last_storage, timeout));
        Com<ID3D12Resource> readback;
        VEL_TRY(make_resource(c, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_FLAG_NONE, readback));
        c->compute_hold.reserve(1);
        VEL_TRY(begin_compute(c, timeout));
        hold(c, readback.p);
        transition(c, b->resource.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        c->list->CopyBufferRegion(readback.p, 0, b->resource.p, offset, bytes);
        transition(c, b->resource.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        VEL_TRY(submit_compute(c, &b->last_compute));
        VEL_TRY(wait_fence(c, c->compute_fence.p, b->last_compute, timeout));
        void *mapped = nullptr;
        D3D12_RANGE read{0, bytes};
        VEL_TRY(readback->Map(0, &read, &mapped));
        std::memcpy(destination, mapped, bytes);
        D3D12_RANGE wrote{0, 0};
        readback->Unmap(0, &wrote);
        c->compute_hold.clear();
        return S_OK;
    } VEL_CATCH
}

extern "C" vel_result vel_release_buffer(vel_context *c, vel_buffer *b, uint32_t timeout) {
    if (!c || !b || b->owner != c) return E_INVALIDARG;
    VEL_TRY(settle(c, c->storage_fence.p, b->last_storage, timeout));
    VEL_TRY(settle(c, c->compute_fence.p, b->last_compute, timeout));
    // Bindless dispatches do not record which buffers they touch.
    if (b->srv_index != UINT32_MAX) VEL_TRY(settle(c, c->compute_fence.p, c->last_bindless_compute, timeout));
    if (b->srv_index != UINT32_MAX) c->heap_free.push_back(b->srv_index);
    if (b->uav_index != UINT32_MAX) c->heap_free.push_back(b->uav_index);
    c->buffers.erase(std::find(c->buffers.begin(), c->buffers.end(), b));
    delete b;
    return S_OK;
}

extern "C" vel_result vel_release_file(vel_context *c, vel_file *f, uint32_t timeout) {
    if (!c || !f || f->owner != c) return E_INVALIDARG;
    VEL_TRY(settle(c, c->storage_fence.p, f->last_storage, timeout));
    c->files.erase(std::find(c->files.begin(), c->files.end(), f));
    delete f;
    return S_OK;
}

extern "C" vel_result vel_destroy(vel_context *c, uint32_t timeout) {
    if (!c) return S_OK;
    VEL_TRY(settle(c, c->storage_fence.p, c->next_storage, timeout));
    VEL_TRY(settle(c, c->compute_fence.p, c->next_compute, timeout));
    c->storage->Close();
    delete c;
    return S_OK;
}

// ---- Bindless compute API ----

static uint32_t alloc_descriptor(vel_context *c) {
    if (!c->heap_free.empty()) {
        const uint32_t index = c->heap_free.back();
        c->heap_free.pop_back();
        return index;
    }
    if (c->heap_next >= c->heap_capacity) return UINT32_MAX;
    return c->heap_next++;
}

static D3D12_CPU_DESCRIPTOR_HANDLE descriptor_cpu(vel_context *c, uint32_t index) {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = cpu_heap_start(c->heap.p);
    handle.ptr += uint64_t(index) * c->descriptor_size;
    return handle;
}

extern "C" vel_result vel_create_resource(vel_context *c, uint64_t bytes, uint32_t kind,
    vel_buffer **out, vel_buffer_info *info) {
    if (!c || !out || !info || !bytes || kind > VEL_RESOURCE_HOST) return E_INVALIDARG;
    if (!c->heap.p) return DXGI_ERROR_UNSUPPORTED;
    *out = nullptr;
    *info = {};
    const uint64_t padded = (bytes + 255) & ~uint64_t(255);
    // Raw views address 32-bit elements; shaders use 32-bit byte offsets.
    if (padded / 4 > UINT32_MAX || padded > (uint64_t(1) << 32)) return E_INVALIDARG;
    try {
        auto b = std::make_unique<vel_buffer>();
        b->owner = c;
        b->bytes = bytes;
        D3D12_HEAP_PROPERTIES heap{};
        if (kind == VEL_RESOURCE_HOST) {
            heap.Type = D3D12_HEAP_TYPE_CUSTOM;
            heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
            heap.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        } else {
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        }
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = padded;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        VEL_TRY(c->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(b->resource.put())));
        if (kind == VEL_RESOURCE_HOST) {
            D3D12_RANGE none{0, 0};
            VEL_TRY(b->resource->Map(0, &none, &b->mapped));
            std::memset(b->mapped, 0, static_cast<size_t>(padded));
        }
        b->srv_index = alloc_descriptor(c);
        b->uav_index = alloc_descriptor(c);
        if (b->srv_index == UINT32_MAX || b->uav_index == UINT32_MAX) {
            if (b->srv_index != UINT32_MAX) c->heap_free.push_back(b->srv_index);
            if (b->uav_index != UINT32_MAX) c->heap_free.push_back(b->uav_index);
            return E_OUTOFMEMORY;
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R32_TYPELESS;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = static_cast<UINT>(padded / 4);
        srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        c->device->CreateShaderResourceView(b->resource.p, &srv, descriptor_cpu(c, b->srv_index));
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = static_cast<UINT>(padded / 4);
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        c->device->CreateUnorderedAccessView(b->resource.p, nullptr, &uav, descriptor_cpu(c, b->uav_index));
        info->srv_index = b->srv_index;
        info->uav_index = b->uav_index;
        info->mapped = b->mapped;
        info->gpu_address = b->resource->GetGPUVirtualAddress();
        info->allocated_bytes = padded;
        c->buffers.push_back(b.get());
        *out = b.release();
        return S_OK;
    } VEL_CATCH
}

extern "C" vel_result vel_create_pipeline(vel_context *c, const void *dxil, size_t bytes, vel_pipeline **out) {
    if (!c || !dxil || !bytes || !out) return E_INVALIDARG;
    if (!c->bindless_root.p) return DXGI_ERROR_UNSUPPORTED;
    *out = nullptr;
    try {
        auto p = std::make_unique<vel_pipeline>();
        p->owner = c;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = c->bindless_root.p;
        desc.CS = {dxil, bytes};
        VEL_TRY(c->device->CreateComputePipelineState(&desc, IID_PPV_ARGS(p->pso.put())));
        c->pipelines.push_back(p.get());
        *out = p.release();
        return S_OK;
    } VEL_CATCH
}

// Loads agility\\dxcompiler.dll from the executable's directory, so its dxil.dll resolves
// beside it rather than to another DXC in the process (Dawn's).
static HRESULT load_dxc(vel_context *c) {
    if (c->dxc.p) return S_OK;
    wchar_t path[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (!n || n >= MAX_PATH) return HRESULT_FROM_WIN32(GetLastError());
    std::wstring dll(path, n);
    dll.resize(dll.find_last_of(L'\\') + 1);
    c->dxc_include_path = dll + L"agility\\include";
    dll += L"agility\\dxcompiler.dll";
    c->dxc_module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!c->dxc_module) {
        c->last_error = "cannot load agility\\dxcompiler.dll";
        return HRESULT_FROM_WIN32(GetLastError());
    }
    auto create = reinterpret_cast<DxcCreateInstanceProc>(
        reinterpret_cast<void*>(GetProcAddress(c->dxc_module, "DxcCreateInstance")));
    if (!create) return HRESULT_FROM_WIN32(GetLastError());
    VEL_TRY(create(CLSID_DxcUtils, IID_PPV_ARGS(c->dxc_utils.put())));
    VEL_TRY(c->dxc_utils->CreateDefaultIncludeHandler(c->dxc_includes.put()));
    return create(CLSID_DxcCompiler, IID_PPV_ARGS(c->dxc.put()));
}

extern "C" vel_result vel_compile_pipeline(vel_context *c, const char *source, size_t bytes,
    const char *target, vel_pipeline **out) {
    if (!c || !source || !bytes || !target || !out) return E_INVALIDARG;
    *out = nullptr;
    try {
        c->last_error.clear();
        VEL_TRY(load_dxc(c));
        std::wstring wtarget(target, target + std::strlen(target));
        const wchar_t *args[] = {L"-E", L"main", L"-T", wtarget.c_str(), L"-HV", L"2021", L"-O3",
            L"-enable-16bit-types", L"-Qstrip_debug", L"-Qstrip_reflect",
            L"-I", c->dxc_include_path.c_str()};
        DxcBuffer buffer{source, bytes, DXC_CP_UTF8};
        Com<IDxcResult> result;
        VEL_TRY(c->dxc->Compile(&buffer, args, sizeof(args) / sizeof(args[0]), c->dxc_includes.p,
            IID_PPV_ARGS(result.put())));
        HRESULT status;
        VEL_TRY(result->GetStatus(&status));
        if (FAILED(status)) {
            Com<IDxcBlobUtf8> errors;
            if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(errors.put()), nullptr)) &&
                errors.p && errors->GetStringLength())
                c->last_error.assign(errors->GetStringPointer(), errors->GetStringLength());
            return status;
        }
        Com<IDxcBlob> object;
        VEL_TRY(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(object.put()), nullptr));
        return vel_create_pipeline(c, object->GetBufferPointer(), object->GetBufferSize(), out);
    } VEL_CATCH
}

extern "C" size_t vel_last_error(vel_context *c, char *out, size_t capacity) {
    if (!c || !out || !capacity) return 0;
    const size_t n = std::min(capacity - 1, c->last_error.size());
    std::memcpy(out, c->last_error.data(), n);
    out[n] = 0;
    return n;
}

extern "C" vel_result vel_remove_device(vel_context *c) {
    if (!c) return E_INVALIDARG;
    Com<ID3D12Device5> device5;
    VEL_TRY(c->device->QueryInterface(IID_PPV_ARGS(device5.put())));
    device5->RemoveDevice();
    return S_OK;
}

extern "C" vel_result vel_release_pipeline(vel_context *c, vel_pipeline *p, uint32_t timeout) {
    if (!c || !p || p->owner != c) return E_INVALIDARG;
    VEL_TRY(settle(c, c->compute_fence.p, c->last_bindless_compute, timeout));
    c->pipelines.erase(std::find(c->pipelines.begin(), c->pipelines.end(), p));
    delete p;
    return S_OK;
}

extern "C" vel_result vel_begin(vel_context *c, uint32_t timeout) {
    if (!c) return E_INVALIDARG;
    if (!c->bindless_root.p) return DXGI_ERROR_UNSUPPORTED;
    if (c->recording >= 0) return E_ILLEGAL_METHOD_CALL;
    Frame &frame = c->frames[c->next_frame];
    VEL_TRY(wait_fence(c, c->compute_fence.p, frame.ticket, timeout));
    VEL_TRY(frame.allocator->Reset());
    VEL_TRY(frame.list->Reset(frame.allocator.p, nullptr));
    ID3D12DescriptorHeap *heaps[] = {c->heap.p};
    frame.list->SetDescriptorHeaps(1, heaps);
    frame.list->SetComputeRootSignature(c->bindless_root.p);
    c->recording = static_cast<int>(c->next_frame);
    c->next_frame = (c->next_frame + 1) % 3;
    c->timestamp_count = 0;
    return S_OK;
}

extern "C" vel_result vel_dispatch(vel_context *c, vel_pipeline *p, const uint32_t *constants,
    uint32_t constant_count, uint32_t x, uint32_t y, uint32_t z) {
    if (!c || !p || p->owner != c || c->recording < 0 || constant_count > bindless_constant_dwords ||
        (constant_count && !constants) || !x || !y || !z || x > 65535 || y > 65535 || z > 65535) return E_INVALIDARG;
    auto *list = c->frames[c->recording].list.p;
    list->SetPipelineState(p->pso.p);
    if (constant_count) list->SetComputeRoot32BitConstants(0, constant_count, constants, 0);
    list->Dispatch(x, y, z);
    return S_OK;
}

extern "C" vel_result vel_barrier(vel_context *c) {
    if (!c || c->recording < 0) return E_INVALIDARG;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;
    c->frames[c->recording].list->ResourceBarrier(1, &barrier);
    return S_OK;
}

extern "C" vel_result vel_timestamp(vel_context *c, uint32_t *out_index) {
    if (!c || !out_index || c->recording < 0) return E_INVALIDARG;
    if (c->timestamp_count >= 4096) return E_OUTOFMEMORY;
    *out_index = c->timestamp_count;
    c->frames[c->recording].list->EndQuery(c->timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, c->timestamp_count++);
    return S_OK;
}

extern "C" vel_result vel_submit(vel_context *c, uint64_t *ticket) {
    if (!c || !ticket || c->recording < 0) return E_INVALIDARG;
    Frame &frame = c->frames[c->recording];
    if (c->timestamp_count)
        frame.list->ResolveQueryData(c->timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, c->timestamp_count,
            c->timestamp_readback.p, 0);
    c->recording = -1;
    VEL_TRY(frame.list->Close());
    if (c->next_compute == UINT64_MAX - 1) return E_UNEXPECTED;
    ID3D12CommandList *lists[] = {frame.list.p};
    c->compute->ExecuteCommandLists(1, lists);
    *ticket = ++c->next_compute;
    frame.ticket = *ticket;
    c->last_bindless_compute = *ticket;
    return c->compute->Signal(c->compute_fence.p, *ticket);
}

extern "C" vel_result vel_wait_compute(vel_context *c, uint64_t ticket, uint32_t timeout) {
    if (!c || ticket > c->next_compute) return E_INVALIDARG;
    return wait_fence(c, c->compute_fence.p, ticket, timeout);
}

extern "C" vel_result vel_read_timestamps(vel_context *c, uint32_t count, uint64_t *out, uint64_t *frequency) {
    if (!c || !out || !frequency || count > 4096) return E_INVALIDARG;
    void *mapped = nullptr;
    D3D12_RANGE read{0, size_t(count) * 8};
    VEL_TRY(c->timestamp_readback->Map(0, &read, &mapped));
    std::memcpy(out, mapped, size_t(count) * 8);
    D3D12_RANGE none{0, 0};
    c->timestamp_readback->Unmap(0, &none);
    *frequency = c->timestamp_frequency;
    return S_OK;
}

extern "C" vel_result vel_queue_wait_storage(vel_context *c, uint64_t storage_ticket) {
    if (!c || storage_ticket > c->next_storage) return E_INVALIDARG;
    return c->compute->Wait(c->storage_fence.p, storage_ticket);
}

extern "C" vel_result vel_read_ranges_after(vel_context *c, vel_file *f, vel_buffer *b,
    const vel_range *ranges, uint32_t count, uint64_t compute_ticket, uint64_t *ticket) {
    if (!c || !f || !b || !ranges || !ticket || f->owner != c || b->owner != c ||
        !count || count > c->queue_capacity / 2 - 2 || compute_ticket > c->next_compute) return E_INVALIDARG;
    *ticket = 0;
    try {
        uint64_t io_complete, compute_complete;
        HRESULT polled = vel_poll(c, &io_complete, &compute_complete);
        if (FAILED(polled) && polled != E_PENDING) return polled;
        if (c->next_storage == UINT64_MAX - 1) return E_UNEXPECTED;
        std::vector<DSTORAGE_REQUEST> requests(count);
        for (uint32_t i = 0; i < count; ++i) {
            const auto &r = ranges[i];
            if (!r.bytes || r.reserved || r.bytes > c->staging_bytes ||
                r.file_offset > f->bytes || r.bytes > f->bytes - r.file_offset ||
                r.buffer_offset > b->bytes || r.bytes > b->bytes - r.buffer_offset) return E_INVALIDARG;
            auto &request = requests[i];
            request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
            request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            request.Source.File = {f->file.p, r.file_offset, r.bytes};
            request.Destination.Buffer = {b->resource.p, r.buffer_offset, r.bytes};
            request.Name = "zllm-arena-range";
        }
        Batch batch{};
        batch.ticket = c->next_storage + 1;
        VEL_TRY(c->storage_factory->CreateStatusArray(1, "zllm-arena-status", IID_PPV_ARGS(batch.status.put())));
        c->batches.push_back(std::move(batch));
        c->storage->EnqueueRequests(requests.data(), count, c->compute_fence.p,
            compute_ticket, DSTORAGE_ENQUEUE_REQUEST_FLAG_FENCE_WAIT_BEFORE_GPU_WORK);
        c->storage->EnqueueStatus(c->batches.back().status.p, 0);
        c->storage->EnqueueSignal(c->storage_fence.p, ++c->next_storage);
        c->storage->Submit();
        b->last_storage = f->last_storage = *ticket = c->next_storage;
        return S_OK;
    } VEL_CATCH
}
