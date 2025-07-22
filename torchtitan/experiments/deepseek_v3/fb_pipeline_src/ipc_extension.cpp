#include <torch/extension.h>

// Conditionally include the correct exception header based on the platform,
// using the custom USE_HIP macro passed from setup.py.
#ifdef USE_HIP
#include <c10/hip/HIPException.h> // For C10_HIP_CHECK and hipIpc... types
#else
#include <c10/cuda/CUDAException.h> // For C10_CUDA_CHECK and cudaIpc... types
#endif


// Define the IPC memory handle structure, which is 64 bytes for both
// CUDA and HIP.
struct IpcMemHandle {
    char reserved[64];
};

// --- C++ Implementation of IPC Handle Operations ---

// Gets a CUDA/HIP IPC memory handle for a given tensor's storage.
py::bytes get_ipc_handle(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on a CUDA/HIP device");

    // The handle corresponds to the base pointer of the storage.
    // In modern PyTorch, .data_ptr() returns a c10::DataPtr, so we use .get()
    // to access the raw void*.
    void* data_ptr = tensor.storage().data_ptr().get();
    IpcMemHandle handle;

    #ifdef USE_HIP
        // AMD HIP Path
        C10_HIP_CHECK(hipIpcGetMemHandle(
            reinterpret_cast<hipIpcMemHandle_t*>(&handle), data_ptr));
    #else
        // NVIDIA CUDA Path
        C10_CUDA_CHECK(cudaIpcGetMemHandle(
            reinterpret_cast<cudaIpcMemHandle_t*>(&handle), data_ptr));
    #endif

    return py::bytes(reinterpret_cast<char*>(&handle), sizeof(handle));
}

// Opens a handle and returns a tensor view on the shared memory.
torch::Tensor open_ipc_handle(
    py::bytes handle_bytes,
    c10::Device device,
    at::ScalarType dtype,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    int64_t storage_offset,
    int64_t storage_nbytes) {

    IpcMemHandle handle;
    // Use py::len() to get the length of a py::bytes object.
    TORCH_CHECK(
        py::len(handle_bytes) == sizeof(handle),
        "IPC handle has incorrect size"
    );
    std::memcpy(&handle, std::string(handle_bytes).c_str(), sizeof(handle));

    void* data_ptr{nullptr};

    #ifdef USE_HIP
        // AMD HIP Path
        C10_HIP_CHECK(hipIpcOpenMemHandle(
            &data_ptr,
            *reinterpret_cast<hipIpcMemHandle_t*>(&handle),
            hipIpcMemLazyEnablePeerAccess));
    #else
        // NVIDIA CUDA Path
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(
            &data_ptr,
            *reinterpret_cast<cudaIpcMemHandle_t*>(&handle),
            cudaIpcMemLazyEnablePeerAccess));
    #endif

    // Use c10::DataPtr and c10::Storage to correctly wrap the raw pointer
    // and manage its lifetime with a custom deleter.
    c10::DataPtr storage_ptr(
        data_ptr,
        data_ptr, // context for deleter
        [](void* ctx) { // deleter lambda
            #ifdef USE_HIP
                C10_HIP_CHECK(hipIpcCloseMemHandle(ctx));
            #else
                C10_CUDA_CHECK(cudaIpcCloseMemHandle(ctx));
            #endif
        },
        device);

    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        storage_nbytes,
        std::move(storage_ptr));

    // Create an empty tensor and use set_() to make it a view on the storage.
    auto final_tensor = torch::empty(
        {0}, torch::TensorOptions().dtype(dtype).device(device));
    final_tensor.set_(storage, storage_offset, shape, stride);
    return final_tensor;
}

// --- pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_ipc_handle", &get_ipc_handle,
          "Get a CUDA/HIP IPC memory handle for a tensor's storage.");
    m.def("open_ipc_handle", &open_ipc_handle,
          "Open a CUDA/HIP IPC handle and return a tensor view.");
}