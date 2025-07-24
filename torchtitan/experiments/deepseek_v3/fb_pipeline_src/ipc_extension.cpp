#include <torch/extension.h>
#include <iostream>
#include <iomanip>
#include <vector>

// Conditionally include the correct exception header based on the platform,
// using the custom USE_HIP macro passed from setup.py.
#ifdef USE_HIP
#include <c10/hip/HIPException.h> // For C10_HIP_CHECK and hipIpc... types
#else
#include <c10/cuda/CUDAException.h> // For C10_CUDA_CHECK and cudaIpc... types
#endif

// #define _PRINT_DEBUG 1

// Define the IPC memory handle structure, which is 64 bytes for both
// CUDA and HIP.
struct IpcMemHandle 
{
    char reserved[64];
};

// Helper function to print bytes from GPU memory for debugging
void print_gpu_bytes(const void* gpu_ptr, size_t nbytes=10)
{
    // Copy 10 bytes from GPU to CPU for printing
    if (nbytes > 10)
    {
        std::cout << "\tFirst 10 IPC memory bytes (hex): ";
        nbytes = 10;
    } 
    else 
    {
        std::cout << "\tIPC memory bytes (hex): ";
    }

    std::vector<char> host_data(nbytes);
    #ifdef USE_HIP
        C10_HIP_CHECK(hipMemcpy(host_data.data(), gpu_ptr, nbytes, hipMemcpyDeviceToHost));
    #else
        C10_CUDA_CHECK(cudaMemcpy(host_data.data(), gpu_ptr, nbytes, cudaMemcpyDeviceToHost));
    #endif

    std::cout << std::hex << std::setfill('0');
    for(int i = 0; i < nbytes; ++i) {
        std::cout << std::setw(2) << static_cast<int>(static_cast<unsigned char>(host_data[i])) << " ";
    }
    std::cout << std::dec << std::endl; // Reset to decimal mode
}

// Helper function to print the raw IPC handle bytes
void print_ipc_handle(const IpcMemHandle& handle) 
{
    std::cout << "\tIPC Handle (64 bytes, hex):";
    std::cout << std::hex << std::setfill('0');
    const auto* ptr = reinterpret_cast<const unsigned char*>(&handle);
    for (size_t i = 0; i < sizeof(handle); ++i) {
        if (i % 16 == 0) std::cout << "\n                             ";
        std::cout << std::setw(2) << static_cast<int>(ptr[i]) << " ";
    }
    std::cout << std::dec << std::endl; // Reset to decimal mode
}


// --- C++ Implementation of IPC Handle Operations ---

py::tuple copy_tensor_and_get_ipc(const torch::Tensor& tensor) 
{
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on a CUDA/HIP device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous for IPC");

    const auto nbytes = tensor.nbytes();
    const auto options = tensor.options();
    void* original_ptr = tensor.data_ptr();
    void* new_gpu_ptr{nullptr};
    
    #ifdef USE_HIP
        C10_HIP_CHECK(hipMalloc(&new_gpu_ptr, nbytes));
        C10_HIP_CHECK(hipMemcpy(new_gpu_ptr, original_ptr, nbytes, hipMemcpyDeviceToDevice));
    #else
        C10_CUDA_CHECK(cudaMalloc(&new_gpu_ptr, nbytes));
        C10_CUDA_CHECK(cudaMemcpy(new_gpu_ptr, original_ptr, nbytes, cudaMemcpyDeviceToDevice));
    #endif

    IpcMemHandle handle;
    #ifdef USE_HIP
        C10_HIP_CHECK(hipIpcGetMemHandle(
            reinterpret_cast<hipIpcMemHandle_t*>(&handle), new_gpu_ptr
        ));
    #else
        C10_CUDA_CHECK(cudaIpcGetMemHandle(
            reinterpret_cast<cudaIpcMemHandle_t*>(&handle), new_gpu_ptr
        ));
    #endif

    auto new_tensor = torch::from_blob(
        new_gpu_ptr,
        tensor.sizes(),
        tensor.strides(),
        [new_gpu_ptr](void*){ 
            #ifdef USE_HIP
                C10_HIP_CHECK(hipFree(new_gpu_ptr));
            #else
                C10_CUDA_CHECK(cudaFree(new_gpu_ptr));
            #endif
        }, // Deleter
        options
    );
    py::bytes handle_bytes(reinterpret_cast<char*>(&handle), sizeof(handle));

    #ifdef _PRINT_DEBUG
    // --- DEBUG PRINTS (CREATE) ---
    auto numel = c10::multiply_integers(tensor.sizes());
    std::cout << "--- [DEBUG IPC COPY (OLD)] ---" << std::endl;
    std::cout << "\ttensor.data_ptr(): " << tensor.data_ptr() << std::endl;
    std::cout << "\ttensor.dtype(): " << tensor.dtype() << std::endl;
    std::cout << "\ttensor.device(): " << tensor.device() << std::endl;
    std::cout << "\ttensor.sizes(): " << tensor.sizes() << std::endl;
    std::cout << "\ttensor.strides(): " << tensor.strides() << std::endl;
    std::cout << "\ttensor.element_size(): " << tensor.element_size() << std::endl;
    print_gpu_bytes(tensor.data_ptr(), numel * tensor.element_size());
    std::cout << "---------------------" << std::endl;
    numel = c10::multiply_integers(new_tensor.sizes());
    std::cout << "--- [DEBUG IPC COPY (NEW)] ---" << std::endl;
    std::cout << "\tnew_tensor.data_ptr(): " << new_tensor.data_ptr() << std::endl;
    std::cout << "\tnew_tensor.dtype(): " << new_tensor.dtype() << std::endl;
    std::cout << "\tnew_tensor.device(): " << new_tensor.device() << std::endl;
    std::cout << "\tnew_tensor.sizes(): " << new_tensor.sizes() << std::endl;
    std::cout << "\tnew_tensor.strides(): " << new_tensor.strides() << std::endl;
    std::cout << "\tnew_tensor.element_size(): " << new_tensor.element_size() << std::endl;
    print_gpu_bytes(new_tensor.data_ptr(), numel * new_tensor.element_size());
    print_ipc_handle(handle);
    std::cout << "---------------------" << std::endl;
    #endif

    return py::make_tuple(handle_bytes, new_tensor);
}

py::tuple create_tensor_and_get_ipc(
    c10::IntArrayRef shape,
    at::ScalarType dtype,
    c10::Device device) 
{
    const auto element_size = c10::elementSize(dtype);
    TORCH_CHECK(element_size > 0, "dtype has invalid element size");

    // Calculate total storage needed
    auto numel = c10::multiply_integers(shape);
    const auto storage_nbytes = numel * element_size;
    const auto storage_offset = 0;  // No offset for new tensors
    
    void* base_ptr{nullptr};

    #ifdef _PRINT_DEBUG
    std::cout << "Creating tensor on device: " << device << std::endl;
    std::cout << "Size: " <<  storage_nbytes << std::endl;
    std::cout << "Shape: " <<  shape << std::endl;
    std::cout << "Element size: " <<  element_size << std::endl;
    std::cout << "Numel: " <<  numel << std::endl;
    std::cout << "Storage offset: " <<  storage_offset << std::endl;
    #endif
    
    #ifdef USE_HIP
        C10_HIP_CHECK(hipMalloc(&base_ptr, storage_nbytes));
    #else
        C10_CUDA_CHECK(cudaMalloc(&base_ptr, storage_nbytes));
    #endif

    void* data_ptr = static_cast<char*>(base_ptr) + storage_offset * element_size;

    auto options = torch::TensorOptions().dtype(dtype).device(device);
    
    // Calculate default stride for contiguous tensor
    std::vector<int64_t> stride(shape.size());
    if (!shape.empty()) {
        stride[shape.size() - 1] = 1;
        for (int64_t i = shape.size() - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    }

    auto tensor = torch::from_blob(
        data_ptr,
        shape,
        stride,
        [base_ptr](void* /*ctx*/) {
            #ifdef USE_HIP
                C10_HIP_CHECK(hipFree(base_ptr));
            #else
                C10_CUDA_CHECK(cudaFree(base_ptr));
            #endif
        },
        options
    );

    // Get IPC handle for the allocated memory
    IpcMemHandle handle;
    #ifdef USE_HIP
        C10_HIP_CHECK(hipIpcGetMemHandle(
            reinterpret_cast<hipIpcMemHandle_t*>(&handle), base_ptr
        ));
    #else
        C10_CUDA_CHECK(cudaIpcGetMemHandle(
            reinterpret_cast<cudaIpcMemHandle_t*>(&handle), base_ptr
        ));
    #endif
    
    #ifdef _PRINT_DEBUG
    // --- DEBUG PRINTS (CREATE) ---
    std::cout << "--- [DEBUG IPC CREATE] ---" << std::endl;
    std::cout << "\ttensor.data_ptr(): " << tensor.data_ptr() << std::endl;
    std::cout << "\ttensor.dtype(): " << tensor.dtype() << std::endl;
    std::cout << "\ttensor.device(): " << tensor.device() << std::endl;
    std::cout << "\ttensor.sizes(): " << tensor.sizes() << std::endl;
    std::cout << "\ttensor.strides(): " << tensor.strides() << std::endl;
    std::cout << "\ttensor.element_size(): " << tensor.element_size() << std::endl;
    print_gpu_bytes(tensor.data_ptr(), numel * tensor.element_size());
    print_ipc_handle(handle);
    std::cout << "---------------------" << std::endl;
    #endif

    py::bytes handle_bytes(reinterpret_cast<char*>(&handle), sizeof(handle));
    return py::make_tuple(handle_bytes, tensor);
}

// Opens a handle and returns a tensor view on the shared memory.
torch::Tensor open_ipc_and_get_tensor(
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
        "IPC handle has incorrect size"
    );
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

    auto tensor = torch::from_blob(
        data_ptr,
        shape,
        stride,
        [base_ptr](void* /*ctx*/) {
            #ifdef USE_HIP
                C10_HIP_CHECK(hipIpcCloseMemHandle(base_ptr));
            #else
                C10_CUDA_CHECK(cudaIpcCloseMemHandle(base_ptr));
            #endif
        },
        options
    );

    #ifdef _PRINT_DEBUG
    // --- DEBUG PRINTS (OPEN) ---
    auto numel = c10::multiply_integers(shape);
    std::cout << "--- [DEBUG IPC GET] ---" << std::endl;
    std::cout << "\ttensor.data_ptr(): " << tensor.data_ptr() << std::endl;
    std::cout << "\ttensor.dtype(): " << tensor.dtype() << std::endl;
    std::cout << "\ttensor.device(): " << tensor.device() << std::endl;
    std::cout << "\ttensor.sizes(): " << tensor.sizes() << std::endl;
    std::cout << "\ttensor.strides(): " << tensor.strides() << std::endl;
    std::cout << "\ttensor.element_size(): " << tensor.element_size() << std::endl;
    print_gpu_bytes(data_ptr, numel * tensor.element_size());
    print_ipc_handle(handle);
    std::cout << "--------------------" << std::endl;
    #endif

    return tensor;
}

// --- pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_tensor_and_get_ipc", &copy_tensor_and_get_ipc,
          "Copy a tensor to new CUDA/HIP memory and return its IPC handle and tensor.");
    m.def("create_tensor_and_get_ipc", &create_tensor_and_get_ipc,
          "Create a new tensor on CUDA/HIP and return its IPC handle and tensor.");
    m.def("open_ipc_and_get_tensor", &open_ipc_and_get_tensor,
          "Open a CUDA/HIP IPC handle and return a tensor view.");
}
