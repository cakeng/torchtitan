#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h> // For at::empty
#include <iostream>
#include <iomanip>
#include <vector>

// Conditionally include the correct exception header and stream headers
// based on the platform.
#ifdef USE_HIP
#include <c10/hip/HIPException.h> // For C10_HIP_CHECK and hipIpc... types
#include <c10/hip/HIPStream.h>    // For c10::hip::getCurrentHIPStream
#else
#include <c10/cuda/CUDAException.h> // For C10_CUDA_CHECK and cudaIpc... types
#include <c10/cuda/CUDAStream.h>    // For c10::cuda::getCurrentCUDAStream
#endif

// #define _PRINT_DEBUG 1

// Define the IPC memory handle structure, which is 64 bytes for both
// CUDA and HIP.
struct IpcMemHandle {
    char reserved[64];
};

// --- Context for the custom allocator ---
struct IpcAllocationContext {
    void* base_ptr;
    IpcMemHandle handle;
};

// --- Custom Allocator for IPC-enabled memory ---
// This allocator calls cuda/hipMalloc directly to ensure each allocation is a
// unique memory region suitable for IPC. This means the allocation will NOT
// be tracked by torch.cuda.memory_allocated().
class IpcAllocator final : public c10::Allocator {
public:
    IpcAllocator() {}
    c10::DataPtr allocate(size_t nbytes) override {
        // Using hip/cudaMalloc directly is essential to guarantee that we get
        // a new, unique memory allocation from the driver, which is required
        // for creating a valid IPC handle.
        void* base_ptr{nullptr};
#ifdef USE_HIP
        C10_HIP_CHECK(hipMalloc(&base_ptr, nbytes));
#else
        C10_CUDA_CHECK(cudaMalloc(&base_ptr, nbytes));
#endif

        auto* ctx = new IpcAllocationContext();
        ctx->base_ptr = base_ptr;

        // Get the IPC handle for the newly allocated memory.
#ifdef USE_HIP
        C10_HIP_CHECK(hipIpcGetMemHandle(
            reinterpret_cast<hipIpcMemHandle_t*>(&ctx->handle), base_ptr));
#else
        C10_CUDA_CHECK(cudaIpcGetMemHandle(
            reinterpret_cast<cudaIpcMemHandle_t*>(&ctx->handle), base_ptr));
#endif

        // Get the specific device for the current context.
#ifdef USE_HIP
        c10::Device actual_device = c10::hip::getCurrentHIPStream().device();
#else
        c10::Device actual_device = c10::cuda::getCurrentCUDAStream().device();
#endif

        // Report the device as 'cuda' for compatibility with the frontend.
        c10::Device reported_device(c10::kCUDA, actual_device.index());

        return {base_ptr, ctx, &ipc_deleter, reported_device};
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
#ifdef USE_HIP
        C10_HIP_CHECK(hipMemcpy(dest, src, count, hipMemcpyDeviceToDevice));
#else
        C10_CUDA_CHECK(cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice));
#endif
    }

private:
    // This deleter function now frees the memory directly.
    static void ipc_deleter(void* ctx_ptr) {
        auto* ctx = static_cast<IpcAllocationContext*>(ctx_ptr);
#ifdef _PRINT_DEBUG
        std::cout << "--- [DEBUG IPC DELETER] ---" << std::endl;
        std::cout << "\tFreeing base_ptr via hip/cudaFree: " << ctx->base_ptr << std::endl;
        std::cout << "--------------------------" << std::endl;
#endif
        // FIXED: Do not throw exceptions from a deleter.
        // Call the function directly and print a warning on failure.
#ifdef USE_HIP
        hipError_t err = hipFree(ctx->base_ptr);
        if (err != hipSuccess) {
            fprintf(stderr, "WARNING: hipFree failed in IpcAllocator::ipc_deleter with error %d\n", err);
        }
#else
        cudaError_t err = cudaFree(ctx->base_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "WARNING: cudaFree failed in IpcAllocator::ipc_deleter with error %d\n", err);
        }
#endif
        delete ctx;
    }
};

// Create a single, global instance of the allocator.
// This ensures all IPC tensors created by this extension share the exact same
// allocator instance, preventing reference counting issues.
static IpcAllocator g_ipc_allocator;


// Helper function to print bytes from GPU memory for debugging
void print_gpu_bytes(const void* gpu_ptr, size_t nbytes = 10) {
    if (nbytes == 0) return;
    size_t display_nbytes = std::min(nbytes, (size_t)10);
    std::cout << "\tFirst " << display_nbytes << " IPC memory bytes (hex): ";

    std::vector<char> host_data(display_nbytes);
#ifdef USE_HIP
    C10_HIP_CHECK(hipMemcpy(host_data.data(), gpu_ptr, display_nbytes, hipMemcpyDeviceToHost));
#else
    C10_CUDA_CHECK(cudaMemcpy(host_data.data(), gpu_ptr, display_nbytes, cudaMemcpyDeviceToHost));
#endif

    std::cout << std::hex << std::setfill('0');
    for (size_t i = 0; i < display_nbytes; ++i) {
        std::cout << std::setw(2) << static_cast<int>(static_cast<unsigned char>(host_data[i])) << " ";
    }
    std::cout << std::dec << std::endl;
}

// Helper function to print the raw IPC handle bytes
void print_ipc_handle(const IpcMemHandle& handle) {
    std::cout << "\tIPC Handle (64 bytes, hex):";
    std::cout << std::hex << std::setfill('0');
    const auto* ptr = reinterpret_cast<const unsigned char*>(&handle);
    for (size_t i = 0; i < sizeof(handle); ++i) {
        if (i % 16 == 0) std::cout << "\n                      ";
        std::cout << std::setw(2) << static_cast<int>(ptr[i]) << " ";
    }
    std::cout << std::dec << std::endl;
}


// --- Create tensor using the custom allocator ---
py::tuple create_tensor_and_get_ipc(
    c10::IntArrayRef shape,
    at::ScalarType dtype,
    c10::Device device)
{
    TORCH_CHECK(device.is_cuda(), "Device must be CUDA/HIP");
    c10::DeviceGuard guard(device);

    const auto element_size = c10::elementSize(dtype);
    TORCH_CHECK(element_size > 0, "dtype has invalid element size");

    auto numel = c10::multiply_integers(shape);
    const auto storage_nbytes = numel * element_size;

    // Use the global allocator instance.
    c10::DataPtr data_ptr = g_ipc_allocator.allocate(storage_nbytes);
    auto* ctx = static_cast<IpcAllocationContext*>(data_ptr.get_context());
    IpcMemHandle handle = ctx->handle;

    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        storage_nbytes,
        std::move(data_ptr),
        &g_ipc_allocator, // Pass pointer to the global allocator
        /*resizable=*/false);

    // Create tensor options using the device from the storage.
    auto options = torch::TensorOptions().dtype(dtype).device(storage.device());
    auto final_tensor = at::empty({0}, options);
    final_tensor.set_(storage, 0, shape, {});

#ifdef _PRINT_DEBUG
    std::cout << "--- [DEBUG IPC CREATE] ---" << std::endl;
    std::cout << "\ttensor.data_ptr(): " << final_tensor.data_ptr() << std::endl;
    std::cout << "\ttensor.sizes(): " << final_tensor.sizes() << std::endl;
    print_gpu_bytes(final_tensor.data_ptr(), numel * element_size);
    print_ipc_handle(handle);
    std::cout << "---------------------" << std::endl;
#endif

    py::bytes handle_bytes(reinterpret_cast<char*>(&handle), sizeof(handle));
    // Return order changed to {tensor, handle}
    return py::make_tuple(final_tensor, handle_bytes);
}

// --- Copy an existing tensor to shared IPC memory ---
// Name changed from copy_to_ipc_tensor
py::tuple copy_tensor_and_get_ipc(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Input tensor must be on a CUDA/HIP device");
    TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous for IPC");
    c10::DeviceGuard guard(tensor.device());

    const auto nbytes = tensor.nbytes();

    // Use the global allocator instance.
    c10::DataPtr data_ptr = g_ipc_allocator.allocate(nbytes);
    auto* ctx = static_cast<IpcAllocationContext*>(data_ptr.get_context());
    IpcMemHandle handle = ctx->handle;

    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        std::move(data_ptr),
        &g_ipc_allocator, // Pass pointer to the global allocator
        /*resizable=*/false);

    // Create new options based on the source tensor, but override the
    // device with the one from the new storage.
    auto options = tensor.options().device(storage.device());
    auto new_tensor = at::empty({0}, options);
    new_tensor.set_(storage, 0, tensor.sizes(), tensor.strides());

#ifdef USE_HIP
    C10_HIP_CHECK(hipMemcpy(new_tensor.data_ptr(), tensor.data_ptr(), nbytes, hipMemcpyDeviceToDevice));
#else
    C10_CUDA_CHECK(cudaMemcpy(new_tensor.data_ptr(), tensor.data_ptr(), nbytes, cudaMemcpyDeviceToDevice));
#endif

#ifdef _PRINT_DEBUG
    std::cout << "--- [DEBUG IPC COPY] ---" << std::endl;
    std::cout << "\tSource tensor ptr: " << tensor.data_ptr() << std::endl;
    std::cout << "\tNew shared tensor ptr: " << new_tensor.data_ptr() << std::endl;
    std::cout << "\tCopied " << nbytes << " bytes." << std::endl;
    print_ipc_handle(handle);
    std::cout << "-----------------------" << std::endl;
#endif

    py::bytes handle_bytes(reinterpret_cast<char*>(&handle), sizeof(handle));
    // Return order changed to {tensor, handle}
    return py::make_tuple(new_tensor, handle_bytes);
}

// --- MODIFIED: This function is for consumers, not owners, of memory ---
py::tuple open_ipc_and_get_tensor(
    py::bytes handle_bytes,
    c10::Device device,
    at::ScalarType dtype,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    int64_t storage_offset)
{
    IpcMemHandle handle;
    TORCH_CHECK(
        py::len(handle_bytes) == sizeof(handle),
        "IPC handle has incorrect size");
    std::memcpy(&handle, PYBIND11_BYTES_AS_STRING(handle_bytes.ptr()), sizeof(handle));

    void* base_ptr{nullptr};
#ifdef USE_HIP
    C10_HIP_CHECK(hipIpcOpenMemHandle(
        &base_ptr,
        *reinterpret_cast<hipIpcMemHandle_t*>(&handle),
        hipIpcMemLazyEnablePeerAccess));
#else
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(
        &base_ptr,
        *reinterpret_cast<cudaIpcMemHandle_t*>(&handle),
        cudaIpcMemLazyEnablePeerAccess));
#endif

    const auto element_size = c10::elementSize(dtype);
    TORCH_CHECK(element_size > 0, "dtype has invalid element size");
    void* data_ptr = static_cast<char*>(base_ptr) + storage_offset * element_size;

    auto options = torch::TensorOptions().dtype(dtype).device(device);

    // Custom deleter for IPC memory handle
    auto ipc_close_deleter = [base_ptr](void* /*unused*/) {
#ifdef USE_HIP
        hipError_t err = hipIpcCloseMemHandle(base_ptr);
        if (err != hipSuccess) {
            fprintf(stderr, "WARNING: hipIpcCloseMemHandle failed in custom deleter with error %d\n", err);
        }
#else
        cudaError_t err = cudaIpcCloseMemHandle(base_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "WARNING: cudaIpcCloseMemHandle failed in custom deleter with error %d\n", err);
        }
#endif
    };

    auto tensor = torch::from_blob(
        data_ptr,
        shape,
        stride,
        ipc_close_deleter,
        options
    );

#ifdef _PRINT_DEBUG
    auto numel = c10::multiply_integers(shape);
    std::cout << "--- [DEBUG IPC GET] ---" << std::endl;
    std::cout << "\ttensor.data_ptr(): " << tensor.data_ptr() << std::endl;
    std::cout << "\ttensor.sizes(): " << tensor.sizes() << std::endl;
    print_gpu_bytes(data_ptr, numel * tensor.element_size());
    print_ipc_handle(handle);
    std::cout << "--------------------" << std::endl;
#endif

    // MODIFIED: Return the tensor and the opened handle pointer.
    return py::make_tuple(tensor, reinterpret_cast<int64_t>(base_ptr));
}

// --- pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_tensor_and_get_ipc", &create_tensor_and_get_ipc,
          "Create a new tensor on CUDA/HIP using a custom IPC-aware "
          "allocator and return its tensor and IPC handle.");
    m.def("copy_tensor_and_get_ipc", &copy_tensor_and_get_ipc,
          "Copies a tensor to new IPC-aware shared memory and returns "
          "the new tensor and its IPC handle.");
    m.def("open_ipc_and_get_tensor", &open_ipc_and_get_tensor,
          "Open a CUDA/HIP IPC handle and return a tensor view AND the handle pointer for manual closing.");
}
