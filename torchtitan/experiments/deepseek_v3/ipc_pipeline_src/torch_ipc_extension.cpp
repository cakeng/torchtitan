#include <torch/extension.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h> // For at::empty
#include <memory>
#include <iostream>
#include <iomanip>
#include <vector>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>
#include <stdexcept>
#include <cstring> // For strerror
#include <cerrno>  // For errno
#include <semaphore.h> // For sem_t

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

// The data structure to be placed in shared memory
struct SharedLockData {
    std::atomic_flag lock_flag = ATOMIC_FLAG_INIT;
};

class SharedLock {
public:
    // Constructor initializes or attaches to a named shared memory segment
    SharedLock(const std::string& name) : name_(name) {
        std::string shm_name = "/" + name;
        int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) throw std::runtime_error(
            "shm_open failed for name " + shm_name + 
            ": " + std::strerror(errno)
        );
        
        ftruncate(fd, sizeof(SharedLockData));
        data_ = static_cast<SharedLockData*>(mmap(
            nullptr, sizeof(SharedLockData),
            PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0
        ));
        close(fd);
        if (data_ == MAP_FAILED) throw std::runtime_error(
            "mmap failed for name " + shm_name + 
            ": " + std::strerror(errno)
        );
        data_->lock_flag.clear();
    }

    // Destructor to unmap the shared memory segment
    ~SharedLock() {
        if (data_ != nullptr && data_ != MAP_FAILED) {
            munmap(data_, sizeof(SharedLockData));
        }
    }

    void acquire() {
        while (data_->lock_flag.test_and_set(std::memory_order_acquire));
    }

    void release() {
        data_->lock_flag.clear(std::memory_order_release);
    }

    static void destroy(const std::string& name) {
        std::string shm_name = "/" + name;
        shm_unlink(shm_name.c_str());
    }

    const std::string& get_name() const { return name_; }

private:
    std::string name_;
    SharedLockData* data_;
};

struct IpcMemHandle {
    char reserved[64];
};

// --- Context for the custom allocator ---
struct IpcAllocationContext {
    void* base_ptr;
    IpcMemHandle handle;
    std::unique_ptr<SharedLock> lock; // Use a unique_ptr for ownership
};

// --- Custom Allocator for IPC-enabled memory ---
class IpcAllocator final : public c10::Allocator {
public:
    IpcAllocator() {}
    c10::DataPtr allocate(size_t nbytes) override {
        void* base_ptr{nullptr};
#ifdef USE_HIP
        C10_HIP_CHECK(hipMalloc(&base_ptr, nbytes));
#else
        C10_CUDA_CHECK(cudaMalloc(&base_ptr, nbytes));
#endif

        auto* ctx = new IpcAllocationContext();
        ctx->base_ptr = base_ptr;

#ifdef USE_HIP
        C10_HIP_CHECK(hipIpcGetMemHandle(
            reinterpret_cast<hipIpcMemHandle_t*>(&ctx->handle), base_ptr));
#else
        C10_CUDA_CHECK(cudaIpcGetMemHandle(
            reinterpret_cast<cudaIpcMemHandle_t*>(&ctx->handle), base_ptr));
#endif

#ifdef USE_HIP
        c10::Device actual_device = c10::hip::getCurrentHIPStream().device();
#else
        c10::Device actual_device = c10::cuda::getCurrentCUDAStream().device();
#endif
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
    static void ipc_deleter(void* ctx_ptr) {
        auto* ctx = static_cast<IpcAllocationContext*>(ctx_ptr);
        if (!ctx) return;
#ifdef _PRINT_DEBUG
        std::cout << "--- [DEBUG IPC DELETER] ---" << std::endl;
        std::cout << "\tFreeing base_ptr via hip/cudaFree: " << ctx->base_ptr << std::endl;
        std::cout << "--------------------------" << std::endl;
#endif

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
        // Explicitly destroy the OS-level shared memory resource
        if (ctx->lock) {
            SharedLock::destroy(ctx->lock->get_name());
        }
        delete ctx;
    }
};

static IpcAllocator g_ipc_allocator;

std::string generate_lock_name_from_handle(const IpcMemHandle& handle) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    const unsigned char* ptr = reinterpret_cast<const unsigned char*>(&handle);
    for (size_t i = 0; i < sizeof(handle); ++i) {
        oss << std::setw(2) << static_cast<int>(ptr[i]);
    }
    return oss.str();
}

IpcAllocationContext* get_ctx_from_tensor(const torch::Tensor& tensor) {
    auto* ctx = static_cast<IpcAllocationContext*>(tensor.storage().data_ptr().get_context());
    TORCH_CHECK(ctx && ctx->lock, "Tensor does not have a valid IPC lock context.");
    return ctx;
}

void synchronize_device(c10::Device device) {
    TORCH_CHECK(device.is_cuda(), "Device must be CUDA/HIP for synchronization");
    c10::DeviceGuard guard(device);
#ifdef USE_HIP
    // Use device-wide synchronization for robustness
    C10_HIP_CHECK(hipDeviceSynchronize());
#else
    // Use device-wide synchronization for robustness
    C10_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

void acquire_lock_for_tensor(const torch::Tensor& tensor) {
    // #ifdef USE_HIP
    //     roctxRangePush("IPC Lock Critical Section");
    // #else
    //     nvtxRangePushA("IPC Lock Critical Section");
    // #endif
    get_ctx_from_tensor(tensor)->lock->acquire();
}

void release_lock_for_tensor_async(const torch::Tensor& tensor) {
    get_ctx_from_tensor(tensor)->lock->release();
    // #ifdef USE_HIP
    //     roctxRangePop();
    // #else
    //     nvtxRangePop();
    // #endif
}

void release_lock_for_tensor(const torch::Tensor& tensor) {
    // Synchronize the stream to ensure all pending operations are complete
    // before the lock is released. This prevents race conditions.
    synchronize_device(tensor.device());
    get_ctx_from_tensor(tensor)->lock->release();
}

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

    c10::DataPtr data_ptr = g_ipc_allocator.allocate(storage_nbytes);
    auto* ctx = static_cast<IpcAllocationContext*>(data_ptr.get_context());
    IpcMemHandle handle = ctx->handle;

    std::string lock_name = generate_lock_name_from_handle(ctx->handle);
    ctx->lock = std::make_unique<SharedLock>(lock_name);

    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        storage_nbytes,
        std::move(data_ptr),
        &g_ipc_allocator,
        /*resizable=*/false);

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
    return py::make_tuple(final_tensor, handle_bytes);
}

py::tuple copy_tensor_and_get_ipc(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Input tensor must be on a CUDA/HIP device");
    TORCH_CHECK(tensor.is_contiguous(), "Input tensor must be contiguous for IPC");
    c10::DeviceGuard guard(tensor.device());

    const auto nbytes = tensor.nbytes();

    c10::DataPtr data_ptr = g_ipc_allocator.allocate(nbytes);
    auto* ctx = static_cast<IpcAllocationContext*>(data_ptr.get_context());
    IpcMemHandle handle = ctx->handle;

    std::string lock_name = generate_lock_name_from_handle(ctx->handle);
    ctx->lock = std::make_unique<SharedLock>(lock_name);

    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        std::move(data_ptr),
        &g_ipc_allocator,
        /*resizable=*/false);

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
    return py::make_tuple(new_tensor, handle_bytes);
}

torch::Tensor open_ipc_and_get_tensor(
    py::bytes handle_bytes,
    c10::Device device,
    at::ScalarType dtype,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    int64_t storage_offset)
{
    IpcMemHandle handle;
    std::memcpy(&handle, PYBIND11_BYTES_AS_STRING(handle_bytes.ptr()), sizeof(handle));

    void* base_ptr{nullptr};
#ifdef USE_HIP
    C10_HIP_CHECK(hipIpcOpenMemHandle(
        &base_ptr, *reinterpret_cast<hipIpcMemHandle_t*>(&handle),
        hipIpcMemLazyEnablePeerAccess));
#else
    C10_CUDA_CHECK(cudaIpcOpenMemHandle(
        &base_ptr, *reinterpret_cast<cudaIpcMemHandle_t*>(&handle),
        cudaIpcMemLazyEnablePeerAccess));
#endif

    auto* ctx = new IpcAllocationContext();
    ctx->base_ptr = base_ptr;
    ctx->handle = handle;
    std::string lock_name = generate_lock_name_from_handle(handle);
    ctx->lock = std::make_unique<SharedLock>(lock_name);

    auto consumer_deleter = [](void* ctx_ptr) {
        auto* local_ctx = static_cast<IpcAllocationContext*>(ctx_ptr);
#ifdef USE_HIP
        hipIpcCloseMemHandle(local_ctx->base_ptr);
#else
        cudaIpcCloseMemHandle(local_ctx->base_ptr);
#endif
        delete local_ctx;
    };

    c10::DataPtr data_ptr(base_ptr, ctx, consumer_deleter, device);

    const auto nbytes = c10::multiply_integers(shape) * c10::elementSize(dtype);

    c10::Storage storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        std::move(data_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);

    auto options = torch::TensorOptions().dtype(dtype).device(storage.device());
    auto final_tensor = at::empty({0}, options);
    final_tensor.set_(storage, storage_offset, shape, stride);

#ifdef _PRINT_DEBUG
    auto numel = c10::multiply_integers(shape);
    std::cout << "--- [DEBUG IPC GET] ---" << std::endl;
    std::cout << "\ttensor.data_ptr(): " << tensor.data_ptr() << std::endl;
    std::cout << "\ttensor.sizes(): " << tensor.sizes() << std::endl;
    print_gpu_bytes(data_ptr, numel * tensor.element_size());
    print_ipc_handle(handle);
    std::cout << "--------------------" << std::endl;
#endif

    return final_tensor;
}

// The data structure to be placed in shared memory.
struct SharedPayload {
    std::atomic<int64_t> value;
    std::atomic_flag mutex_lock = ATOMIC_FLAG_INIT;
    sem_t semaphore;
};

class SharedData {
public:
    // Constructor: Creates or opens a named shared memory segment.
    SharedData(const std::string& name, bool is_creator, int64_t initial_value, unsigned int semaphore_count) : name_(name) {
        std::string shm_name = "/" + name;
        int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            throw std::runtime_error("shm_open failed for " + shm_name + ": " + std::strerror(errno));
        }
        
        ftruncate(fd, sizeof(SharedPayload));
        data_ = static_cast<SharedPayload*>(mmap(
            nullptr, sizeof(SharedPayload),
            PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0
        ));
        close(fd);

        if (data_ == MAP_FAILED) {
            throw std::runtime_error("mmap failed for " + shm_name + ": " + std::strerror(errno));
        }

        // Only the creator process should initialize the shared data.
        if (is_creator) {
            data_->value.store(initial_value);
            data_->mutex_lock.clear(); 
            // The second argument '1' makes the semaphore shared between processes.
            if (sem_init(&data_->semaphore, 1, semaphore_count) == -1) {
                throw std::runtime_error("sem_init failed: " + std::string(std::strerror(errno)));
            }
        }
    }

    // Destructor unmaps the shared memory from this process's address space.
    ~SharedData() {
        if (data_ != nullptr && data_ != MAP_FAILED) {
            munmap(data_, sizeof(SharedPayload));
        }
    }

    // --- Atomic Integer Operations ---
    int64_t add(int64_t n) { return data_->value.fetch_add(n) + n; }
    int64_t get() { return data_->value.load(); }
    void set(int64_t n) { data_->value.store(n); }
    int64_t fetch_add(int64_t n) { return data_->value.fetch_add(n); }
    int64_t exchange(int64_t n) { return data_->value.exchange(n); }

    // --- Mutex Operations ---
    void mutex_acquire() { while (data_->mutex_lock.test_and_set(std::memory_order_acquire)); }
    void mutex_release() { data_->mutex_lock.clear(std::memory_order_release); }

    // --- Semaphore Operations ---
    void sem_wait() { 
        // FIX: Use :: to call the global POSIX function, not the member function.
        ::sem_wait(&data_->semaphore); 
    }
    void sem_post() { 
        // FIX: Use :: to call the global POSIX function, not the member function.
        ::sem_post(&data_->semaphore); 
    }
    void wait_for_value(int64_t target_value, int poll_interval_us) {
        while (get() != target_value) {
            std::this_thread::sleep_for(std::chrono::microseconds(poll_interval_us));
        }
    }

    // --- Resource Management ---
    static void destroy(const std::string& name) {
        std::string shm_name = "/" + name;
        int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
        if (fd != -1) {
            SharedPayload* temp_data = static_cast<SharedPayload*>(mmap(
                nullptr, sizeof(SharedPayload), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0
            ));
            if (temp_data != MAP_FAILED) {
                sem_destroy(&temp_data->semaphore);
                munmap(temp_data, sizeof(SharedPayload));
            }
            close(fd);
        }
        shm_unlink(shm_name.c_str());
    }

private:
    std::string name_;
    SharedPayload* data_;
};

// --- pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "A C++ PyTorch extension for creating and sharing CUDA/HIP tensors using IPC";
    m.def("create_tensor_and_get_ipc", &create_tensor_and_get_ipc,
          "Create a new tensor on CUDA/HIP using a custom IPC-aware "
          "allocator and return its tensor and IPC handle.");
    m.def("copy_tensor_and_get_ipc", &copy_tensor_and_get_ipc,
          "Copies a tensor to new IPC-aware shared memory and returns "
          "the new tensor and its IPC handle.");
    m.def("open_ipc_and_get_tensor", &open_ipc_and_get_tensor,
          "Open a CUDA/HIP IPC handle and returns a tensor.");
    m.def("acquire", &acquire_lock_for_tensor, "Acquire the lock for a given IPC tensor.");
    m.def("release", &release_lock_for_tensor, 
          "Release the lock for a given IPC tensor, includes device-wide synchronization to avoid race conditions on kernel launched in the critical section.");
    m.def("release_async", &release_lock_for_tensor_async, 
          "Release the lock for a given IPC tensor, Pytorch-level GPU synchronization is required to avoid race conditions on kernel launched in the critical section.");

    py::class_<SharedData>(m, "SharedData")
        .def(py::init<const std::string&, bool, int64_t, unsigned int>(), 
             "Creates or opens a shared data object.",
             py::arg("name"), py::arg("is_creator"), 
             py::arg("initial_value") = 0, py::arg("semaphore_count") = 1)
        
        .def("add", &SharedData::add, "Atomically adds a value and returns the new value.")
        .def("get", &SharedData::get, "Atomically gets the current value.")
        .def("set", &SharedData::set, "Atomically sets the value.")
        .def("fetch_add", &SharedData::fetch_add, "Atomically adds a value and returns the old value.")
        .def("exchange", &SharedData::exchange, "Atomically sets a new value and returns the old value.")

        .def("mutex_acquire", &SharedData::mutex_acquire, "Acquires the exclusive mutex lock.")
        .def("mutex_release", &SharedData::mutex_release, "Releases the exclusive mutex lock.")

        .def("sem_wait", &SharedData::sem_wait, "Waits on (decrements) the semaphore.")
        .def("sem_post", &SharedData::sem_post, "Posts to (increments) the semaphore.")
        .def("wait_for_value", &SharedData::wait_for_value, 
             "Waits until the shared integer reaches a target value.",
             py::arg("target_value"), py::arg("poll_interval_us") = 100);

    m.def("destroy_shared_data", &SharedData::destroy, 
          "Destroys and unlinks the shared memory and semaphore for a SharedData object.",
          py::arg("name"));
}
