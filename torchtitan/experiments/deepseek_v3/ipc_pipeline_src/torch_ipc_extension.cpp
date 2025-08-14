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
#include <pthread.h>
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
struct SharedLockData
{
    pthread_mutex_t mutex;
    std::atomic<int64_t> is_initialized{0};
};

/**
 * @brief A robust, inter-process lock using a pthread mutex.
 *
 * This class provides a simple acquire/release lock interface but uses a
 * kernel-managed pthread_mutex_t as its backend. This is highly efficient
 * for general-purpose locking as it puts waiting processes to sleep,
 * consuming zero CPU, unlike a spinlock which busy-waits.
 */
class SharedLock
{
public:
    /**
     * @brief Constructs or attaches to a named shared lock.
     *
     * This constructor handles both creation and attachment. The first
     * process to call it will create and initialize the shared memory
     * and the mutex. Subsequent processes will attach to the existing object.
     *
     * @param name A unique name for the shared memory segment.
     * @param is_creator Whether this process should create the shared memory.
     */
    SharedLock(const std::string& name, bool is_creator)
        : name_(name)
    {
        std::string shm_name = "/" + name;
        int fd = -1;
        
        if (is_creator)
        {
            fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
            if (fd == -1)
            {
                throw std::runtime_error("shm_open failed for name " +
                                         shm_name + ": " + std::strerror(errno));
            }
            
            // Set the size of the shared memory object.
            if (ftruncate(fd, sizeof(SharedLockData)) == -1)
            {
                close(fd);
                throw std::runtime_error("ftruncate failed for " + shm_name + ": " +
                                       std::string(std::strerror(errno)));
            }
        }
        else
        {
            while (true)
            {
                fd = shm_open(shm_name.c_str(), O_RDWR, 0);
                if (fd != -1)
                {
                    struct stat file_stat;
                    if (fstat(fd, &file_stat) == 0 && file_stat.st_size >= sizeof(SharedLockData))
                    {
                        break;
                    }
                    close(fd); // Close the handle if size is wrong and retry.
                }
                // Wait a moment before retrying to avoid burning the CPU.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        data_ = static_cast<SharedLockData*>(mmap(
            nullptr, sizeof(SharedLockData),
            PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0
        ));
        
        // The file descriptor is no longer needed after mmap.
        if (close(fd) == -1)
        {
            if (data_ != MAP_FAILED)
            {
                munmap(data_, sizeof(SharedLockData));
            }
            throw std::runtime_error("close failed for " + shm_name + ": " +
                                   std::string(std::strerror(errno)));
        }

        if (data_ == MAP_FAILED)
        {
            throw std::runtime_error("mmap failed for name " +
                                     shm_name + ": " + std::string(std::strerror(errno)));
        }
        
        // Initialize the mutex with process-shared attributes.
        if (is_creator)
        {
            pthread_mutexattr_t mutex_attr;
            pthread_mutexattr_init(&mutex_attr);
            pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
            pthread_mutex_init(&data_->mutex, &mutex_attr);
            pthread_mutexattr_destroy(&mutex_attr);
        }

        if (is_creator)
        {
            data_->is_initialized.store(1, std::memory_order_relaxed);
            // printf("SharedLock initialized creator: %s\n", name_.c_str());
        }
        else
        {
            while (data_->is_initialized.load(std::memory_order_relaxed) == 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            // printf("SharedLock initialized client: %s\n", name_.c_str());
        }
    }

    /**
     * @brief Unmaps the shared memory segment from the process.
     */
    ~SharedLock()
    {
        if (data_ != nullptr && data_ != MAP_FAILED)
        {
            munmap(data_, sizeof(SharedLockData));
        }
    }

    /**
     * @brief Acquires the lock. Blocks efficiently if the lock is held.
     */
    void acquire()
    {
        if (pthread_mutex_lock(&data_->mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_lock failed: " +
                                   std::string(std::strerror(errno)));
        }
    }

    /**
     * @brief Releases the lock.
     */
    void release()
    {
        if (pthread_mutex_unlock(&data_->mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_unlock failed: " +
                                   std::string(std::strerror(errno)));
        }
    }

    /**
     * @brief Destroys the mutex and unlinks the shared memory segment.
     *
     * @param name The unique name of the shared memory segment.
     */
    static void destroy(const std::string& name)
    {
        std::string shm_name = "/" + name;
        int fd = shm_open(shm_name.c_str(), O_RDWR, 0);
        if (fd != -1)
        {
            SharedLockData* temp_data =
                static_cast<SharedLockData*>(mmap(
                    nullptr, sizeof(SharedLockData),
                    PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0
            ));

            if (temp_data != MAP_FAILED)
            {
                if (pthread_mutex_destroy(&temp_data->mutex) != 0)
                {
                    fprintf(stderr, "Warning: pthread_mutex_destroy failed: %s\n", 
                           std::strerror(errno));
                }
                
                if (munmap(temp_data, sizeof(SharedLockData)) == -1)
                {
                    fprintf(stderr, "Warning: munmap failed: %s\n", 
                        std::strerror(errno));
                }
            }
            else
            {
                fprintf(stderr, "Warning: mmap failed during cleanup: %s\n", 
                    std::strerror(errno));
            }
            
            if (close(fd) == -1)
            {
                fprintf(stderr, "Warning: close failed during cleanup: %s\n", 
                    std::strerror(errno));
            }
        }
        
        if (shm_unlink(shm_name.c_str()) == -1)
        {
            fprintf(stderr, "Warning: shm_unlink failed: %s\n", std::strerror(errno));
        }
    }

    /**
     * @brief Returns the name of the shared lock.
     */
    const std::string& get_name() const
    {
        return name_;
    }

private:
    std::string name_;
    SharedLockData* data_;
};

struct IpcMemHandle
{
    char reserved[64];
};

// --- Context for the custom allocator ---
struct IpcAllocationContext
{
    void* base_ptr;
    IpcMemHandle handle;
    std::unique_ptr<SharedLock> lock; // Use a unique_ptr for ownership
};

// --- Custom Allocator for IPC-enabled memory ---
class IpcAllocator final : public c10::Allocator
{
public:
    IpcAllocator() {}
    
    c10::DataPtr allocate(size_t nbytes) override
    {
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
        c10::Device actual_device = 
            c10::hip::getCurrentHIPStream().device();
#else
        c10::Device actual_device = 
            c10::cuda::getCurrentCUDAStream().device();
#endif
        c10::Device reported_device(c10::kCUDA, actual_device.index());

        return {base_ptr, ctx, &ipc_deleter, reported_device};
    }

    void copy_data(void* dest, const void* src, 
                   std::size_t count) const override
    {
#ifdef USE_HIP
        C10_HIP_CHECK(hipMemcpy(dest, src, count, hipMemcpyDeviceToDevice));
#else
        C10_CUDA_CHECK(cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice));
#endif
    }

private:
    static void ipc_deleter(void* ctx_ptr)
    {
        auto* ctx = static_cast<IpcAllocationContext*>(ctx_ptr);
        if (!ctx) return;
#ifdef _PRINT_DEBUG
        std::cout << "--- [DEBUG IPC DELETER] ---" << std::endl;
        std::cout << "\tFreeing base_ptr via hip/cudaFree: " 
                  << ctx->base_ptr << std::endl;
        std::cout << "--------------------------" << std::endl;
#endif

#ifdef USE_HIP
        hipError_t err = hipFree(ctx->base_ptr);
        if (err != hipSuccess)
        {
            fprintf(stderr, 
                "WARNING: hipFree failed in IpcAllocator::ipc_deleter "
                "with error %d\n", err);
        }
#else
        cudaError_t err = cudaFree(ctx->base_ptr);
        if (err != cudaSuccess)
        {
            fprintf(stderr, 
                "WARNING: cudaFree failed in IpcAllocator::ipc_deleter "
                "with error %d\n", err);
        }
#endif
        // Explicitly destroy the OS-level shared memory resource
        if (ctx->lock)
        {
            SharedLock::destroy(ctx->lock->get_name());
        }
        delete ctx;
    }
};

static IpcAllocator g_ipc_allocator;

std::string generate_lock_name_from_handle(const IpcMemHandle& handle)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    const unsigned char* ptr = 
        reinterpret_cast<const unsigned char*>(&handle);
    for (size_t i = 0; i < sizeof(handle); ++i)
    {
        oss << std::setw(2) << static_cast<int>(ptr[i]);
    }
    return oss.str();
}

IpcAllocationContext* get_ctx_from_tensor(const torch::Tensor& tensor)
{
    auto* ctx = static_cast<IpcAllocationContext*>(
        tensor.storage().data_ptr().get_context());
    TORCH_CHECK(ctx && ctx->lock, 
                "Tensor does not have a valid IPC lock context.");
    return ctx;
}

void synchronize_device(c10::Device device)
{
    TORCH_CHECK(device.is_cuda(), 
                "Device must be CUDA/HIP for synchronization");
    c10::DeviceGuard guard(device);
#ifdef USE_HIP
    // Use device-wide synchronization for robustness
    C10_HIP_CHECK(hipDeviceSynchronize());
#else
    // Use device-wide synchronization for robustness
    C10_CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

void acquire_lock_for_tensor(const torch::Tensor& tensor)
{
    // #ifdef USE_HIP
    //     roctxRangePush("IPC Lock Critical Section");
    // #else
    //     nvtxRangePushA("IPC Lock Critical Section");
    // #endif
    get_ctx_from_tensor(tensor)->lock->acquire();
}

void release_lock_for_tensor_async(const torch::Tensor& tensor)
{
    get_ctx_from_tensor(tensor)->lock->release();
    // #ifdef USE_HIP
    //     roctxRangePop();
    // #else
    //     nvtxRangePop();
    // #endif
}

void release_lock_for_tensor(const torch::Tensor& tensor)
{
    // Synchronize the stream to ensure all pending operations are complete
    // before the lock is released. This prevents race conditions.
    synchronize_device(tensor.device());
    get_ctx_from_tensor(tensor)->lock->release();
}

// Helper function to print bytes from GPU memory for debugging
void print_gpu_bytes(const void* gpu_ptr, size_t nbytes = 10)
{
    if (nbytes == 0) return;
    size_t display_nbytes = std::min(nbytes, (size_t)10);
    std::cout << "\tFirst " << display_nbytes 
              << " IPC memory bytes (hex): ";

    std::vector<char> host_data(display_nbytes);
#ifdef USE_HIP
    C10_HIP_CHECK(hipMemcpy(host_data.data(), gpu_ptr, display_nbytes, 
                            hipMemcpyDeviceToHost));
#else
    C10_CUDA_CHECK(cudaMemcpy(host_data.data(), gpu_ptr, display_nbytes, 
                              cudaMemcpyDeviceToHost));
#endif

    std::cout << std::hex << std::setfill('0');
    for (size_t i = 0; i < display_nbytes; ++i)
    {
        std::cout << std::setw(2) 
                  << static_cast<int>(static_cast<unsigned char>(host_data[i])) 
                  << " ";
    }
    std::cout << std::dec << std::endl;
}

// Helper function to print the raw IPC handle bytes
void print_ipc_handle(const IpcMemHandle& handle)
{
    std::cout << "\tIPC Handle (64 bytes, hex):";
    std::cout << std::hex << std::setfill('0');
    const auto* ptr = reinterpret_cast<const unsigned char*>(&handle);
    for (size_t i = 0; i < sizeof(handle); ++i)
    {
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
    ctx->lock = std::make_unique<SharedLock>(lock_name, true); // Creator

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
    std::cout << "\ttensor.data_ptr(): " << final_tensor.data_ptr() 
              << std::endl;
    std::cout << "\ttensor.sizes(): " << final_tensor.sizes() << std::endl;
    print_gpu_bytes(final_tensor.data_ptr(), numel * element_size);
    print_ipc_handle(handle);
    std::cout << "---------------------" << std::endl;
#endif

    py::bytes handle_bytes(reinterpret_cast<char*>(&handle), sizeof(handle));
    return py::make_tuple(final_tensor, handle_bytes);
}

py::tuple copy_tensor_and_get_ipc(const torch::Tensor& tensor)
{
    TORCH_CHECK(tensor.is_cuda(), 
                "Input tensor must be on a CUDA/HIP device");
    TORCH_CHECK(tensor.is_contiguous(), 
                "Input tensor must be contiguous for IPC");
    c10::DeviceGuard guard(tensor.device());

    const auto nbytes = tensor.nbytes();

    c10::DataPtr data_ptr = g_ipc_allocator.allocate(nbytes);
    auto* ctx = static_cast<IpcAllocationContext*>(data_ptr.get_context());
    IpcMemHandle handle = ctx->handle;

    std::string lock_name = generate_lock_name_from_handle(ctx->handle);
    ctx->lock = std::make_unique<SharedLock>(lock_name, true); // Creator

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
    C10_HIP_CHECK(hipMemcpy(new_tensor.data_ptr(), tensor.data_ptr(), 
                            nbytes, hipMemcpyDeviceToDevice));
#else
    C10_CUDA_CHECK(cudaMemcpy(new_tensor.data_ptr(), tensor.data_ptr(), 
                              nbytes, cudaMemcpyDeviceToDevice));
#endif

#ifdef _PRINT_DEBUG
    std::cout << "--- [DEBUG IPC COPY] ---" << std::endl;
    std::cout << "\tSource tensor ptr: " << tensor.data_ptr() << std::endl;
    std::cout << "\tNew shared tensor ptr: " << new_tensor.data_ptr() 
              << std::endl;
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
    std::memcpy(&handle, PYBIND11_BYTES_AS_STRING(handle_bytes.ptr()), 
                sizeof(handle));

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
    ctx->lock = std::make_unique<SharedLock>(lock_name, false); // Attacher

    auto consumer_deleter = [](void* ctx_ptr)
    {
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

struct SharedPayload
{
    std::atomic<int64_t> is_initialized{0};

    // --- For the pthread Barrier ---
    pthread_barrier_t barrier;

    // --- For Condition Variable-based waiting ---
    pthread_mutex_t value_mutex;
    pthread_cond_t  value_cond;
    std::atomic<int64_t> value{0};

    // --- For general-purpose semaphore (kept for API compatibility) ---
    sem_t generic_semaphore;
    pthread_mutex_t generic_mutex;

    // --- For generic data exchange ---
    pthread_mutex_t data_buffer_mutex;
    size_t data_size{0};
    size_t data_buffer_size{0};
    alignas(8) char data_buffer[]; // Using struct hack for dynamic size
};

// --- A robust, process-shared data class using pthreads ---
class SharedData
{
public:
    // Constructor: Creates or opens a named shared memory segment.
    SharedData(const std::string& name, bool is_creator, 
               unsigned int barrier_size, int init_value,
               unsigned int semaphore_count, size_t data_buffer_size)
        : name_(name)
    {
        std::string shm_name = "/" + name;
        total_size_ = sizeof(SharedPayload) + data_buffer_size;
        int fd = -1;
        
        if (is_creator)
        {
            fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
            if (fd == -1)
            {
                throw std::runtime_error("shm_open failed for " + shm_name + 
                                    ": " + std::strerror(errno));
            }
            // Set the size of the shared memory object.
            if (ftruncate(fd, total_size_) == -1)
            {
                close(fd);
                throw std::runtime_error("ftruncate failed for " + shm_name + ": " +
                                       std::string(std::strerror(errno)));
            }
            // printf("Creating shared memory object: %s, is_creator: %d, barrier_size: %d, init_value: %d, semaphore_count: %d\n", 
            //     name.c_str(), is_creator, barrier_size, init_value, semaphore_count);
        }
        else
        {
            while (true)
            {
                fd = shm_open(shm_name.c_str(), O_RDWR, 0);
                if (fd != -1)
                {
                    struct stat file_stat;
                    if (fstat(fd, &file_stat) == 0 && 
                        file_stat.st_size >= total_size_)
                    {
                        break;
                    }
                    close(fd); // Close the handle if size is wrong and retry.
                }
                // Wait a moment before retrying to avoid burning the CPU.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        data_ = static_cast<SharedPayload*>(mmap(
            nullptr, total_size_,
            PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0
        ));
        
        if (close(fd) == -1)
        {
            if (data_ != MAP_FAILED)
            {
                munmap(data_, total_size_);
            }
            throw std::runtime_error("close failed for " + shm_name + ": " +
                                   std::string(std::strerror(errno)));
        }

        if (data_ == MAP_FAILED)
        {
            throw std::runtime_error("mmap failed for " + shm_name + ": " +
                                   std::string(std::strerror(errno)));
        }

        // Only the creator process should initialize the shared data.
        if (is_creator)
        {
            data_->data_buffer_size = data_buffer_size;
            data_->value.store(init_value, std::memory_order_relaxed);
            // Initialize the general-purpose semaphore
            if (sem_init(&data_->generic_semaphore, 1, semaphore_count) == -1)
            {
                throw std::runtime_error("sem_init failed: " +
                                       std::string(std::strerror(errno)));
            }

            pthread_barrierattr_t barrier_attr;
            if (pthread_barrierattr_init(&barrier_attr) != 0)
            {
                throw std::runtime_error("pthread_barrierattr_init failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED) != 0)
            {
                pthread_barrierattr_destroy(&barrier_attr);
                throw std::runtime_error("pthread_barrierattr_setpshared failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_barrier_init(&data_->barrier, &barrier_attr, barrier_size) != 0)
            {
                pthread_barrierattr_destroy(&barrier_attr);
                throw std::runtime_error("pthread_barrier_init failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_barrierattr_destroy(&barrier_attr) != 0)
            {
                throw std::runtime_error("pthread_barrierattr_destroy failed: " +
                                       std::string(std::strerror(errno)));
            }
            
            pthread_condattr_t cond_attr;
            if (pthread_condattr_init(&cond_attr) != 0)
            {
                throw std::runtime_error("pthread_condattr_init failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED) != 0)
            {
                throw std::runtime_error("pthread_condattr_setpshared failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_cond_init(&data_->value_cond, &cond_attr) != 0)
            {
                throw std::runtime_error("pthread_cond_init failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_condattr_destroy(&cond_attr) != 0)
            {
                throw std::runtime_error("pthread_condattr_destroy failed: " +
                                       std::string(std::strerror(errno)));
            }
            
            pthread_mutexattr_t mutex_attr;
            if (pthread_mutexattr_init(&mutex_attr) != 0)
            {
                throw std::runtime_error("pthread_mutexattr_init failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED) != 0)
            {
                pthread_mutexattr_destroy(&mutex_attr);
                throw std::runtime_error("pthread_mutexattr_setpshared failed: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_mutex_init(&data_->data_buffer_mutex, &mutex_attr) != 0)
            {
                pthread_mutexattr_destroy(&mutex_attr);
                throw std::runtime_error("pthread_mutex_init failed for data_buffer_mutex: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_mutex_init(&data_->value_mutex, &mutex_attr) != 0)
            {
                pthread_mutexattr_destroy(&mutex_attr);
                throw std::runtime_error("pthread_mutex_init failed for value_mutex: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_mutex_init(&data_->generic_mutex, &mutex_attr) != 0)
            {
                pthread_mutexattr_destroy(&mutex_attr);
                throw std::runtime_error("pthread_mutex_init failed for generic_mutex: " +
                                       std::string(std::strerror(errno)));
            }
            if (pthread_mutexattr_destroy(&mutex_attr) != 0)
            {
                throw std::runtime_error("pthread_mutexattr_destroy failed: " +
                                       std::string(std::strerror(errno)));
            }
        }

        // Wait for all processes to reach this point.
        if (is_creator)
        {
            data_->is_initialized.store(1, std::memory_order_relaxed);
            // printf("SharedData initialized creator: %s\n", name_.c_str());
        }
        else
        {
            while (data_->is_initialized.load(std::memory_order_relaxed) == 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // printf("SharedData initialized client: %s\n", name_.c_str());
        }
    }

    // Destructor is a no-op; cleanup is handled by the static destroy method.
    ~SharedData()
    {
        if (data_ != nullptr && data_ != MAP_FAILED)
        {
            if (munmap(data_, total_size_) == -1)
            {
                fprintf(stderr, "Warning: munmap failed in destructor: %s\n", 
                       std::strerror(errno));
            }
        }
    }

    // --- Barrier ---
    void barrier()
    {
        // This is the correct, standard way to wait at a barrier.
        // It blocks until all 'barrier_size' processes have called it.
        // PTHREAD_BARRIER_SERIAL_THREAD is returned to one arbitrary process,
        // 0 to others. 
        int result = pthread_barrier_wait(&data_->barrier);
        if (result != 0 && result != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            throw std::runtime_error("pthread_barrier_wait failed: " +
                                   std::string(std::strerror(result)));
        }
    }

    // --- Atomic Integer Operations (using condition variables) ---
    void set(int64_t n)
    {
        if (pthread_mutex_lock(&data_->value_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_lock failed: " +
                                   std::string(std::strerror(errno)));
        }
        data_->value.store(n, std::memory_order_relaxed);
        if (pthread_cond_broadcast(&data_->value_cond) != 0)
        {
            pthread_mutex_unlock(&data_->value_mutex);
            throw std::runtime_error("pthread_cond_broadcast failed: " +
                                   std::string(std::strerror(errno)));
        }
        if (pthread_mutex_unlock(&data_->value_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_unlock failed: " +
                                   std::string(std::strerror(errno)));
        }
    }

    int64_t add(int64_t n)
    {
        if (pthread_mutex_lock(&data_->value_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_lock failed: " +
                                   std::string(std::strerror(errno)));
        }
        int64_t new_val = data_->value.fetch_add(n) + n;
        if (pthread_cond_broadcast(&data_->value_cond) != 0)
        {
            pthread_mutex_unlock(&data_->value_mutex);
            throw std::runtime_error("pthread_cond_broadcast failed: " +
                                   std::string(std::strerror(errno)));
        }
        if (pthread_mutex_unlock(&data_->value_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_unlock failed: " +
                                   std::string(std::strerror(errno)));
        }
        return new_val;
    }

    int64_t get()
    {
        return data_->value.load(std::memory_order_relaxed);
    }

    // The definitive, correct wait_for_value implementation.
    void wait_for_value(int64_t target_value)
    {
        if (pthread_mutex_lock(&data_->value_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_lock failed: " +
                                   std::string(std::strerror(errno)));
        }
        while (data_->value.load(std::memory_order_relaxed) != target_value)
        {
            // Atomically unlocks the mutex and waits. Re-locks upon waking.
            int result = pthread_cond_wait(&data_->value_cond, &data_->value_mutex);
            if (result != 0)
            {
                pthread_mutex_unlock(&data_->value_mutex);
                throw std::runtime_error("pthread_cond_wait failed: " +
                                       std::string(std::strerror(result)));
            }
        }
        if (pthread_mutex_unlock(&data_->value_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_unlock failed: " +
                                   std::string(std::strerror(errno)));
        }
    }

    // --- Generic Data Buffer ---
    void write_data(const std::vector<char>& buffer)
    {
        if (buffer.size() > data_->data_buffer_size)
        {
            throw std::runtime_error("Data size exceeds shared buffer capacity. " +
                                   std::to_string(buffer.size()) + " > " +
                                   std::to_string(data_->data_buffer_size));
        }
        if (pthread_mutex_lock(&data_->data_buffer_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_lock failed: " +
                                   std::string(std::strerror(errno)));
        }
        std::memcpy(data_->data_buffer, buffer.data(), buffer.size());
        data_->data_size = buffer.size();
        if (pthread_mutex_unlock(&data_->data_buffer_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_unlock failed: " +
                                   std::string(std::strerror(errno)));
        }
    }

    std::vector<char> read_data()
    {
        if (pthread_mutex_lock(&data_->data_buffer_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_lock failed: " +
                                   std::string(std::strerror(errno)));
        }
        std::vector<char> buffer(data_->data_size);
        std::memcpy(buffer.data(), data_->data_buffer, data_->data_size);
        if (pthread_mutex_unlock(&data_->data_buffer_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_unlock failed: " +
                                   std::string(std::strerror(errno)));
        }
        return buffer;
    }

    // --- General-Purpose Semaphore (for API compatibility) ---
    void sem_wait()
    {
        if (::sem_wait(&data_->generic_semaphore) == -1)
        {
            throw std::runtime_error("sem_wait failed: " +
                                   std::string(std::strerror(errno)));
        }
    }
    
    void sem_post()
    {
        if (::sem_post(&data_->generic_semaphore) == -1)
        {
            throw std::runtime_error("sem_post failed: " +
                                   std::string(std::strerror(errno)));
        }
    }

    // --- Mutex ---
    void mutex_acquire()
    {
        if (pthread_mutex_lock(&data_->generic_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_lock failed: " +
                                   std::string(std::strerror(errno)));
        }
    }
    
    void mutex_release()
    {
        if (pthread_mutex_unlock(&data_->generic_mutex) != 0)
        {
            throw std::runtime_error("pthread_mutex_unlock failed: " +
                                   std::string(std::strerror(errno)));
        }
    }
    
    std::string get_name() const
    {
        return name_;
    }

    // --- Resource Management ---
    void destroy()
    {
        std::string shm_name = "/" + name_;
        // The creator process should be responsible for destroying the objects
        // before unlinking the shared memory. We open it to get a handle.
        int destroy_result = 0;
        destroy_result |= pthread_barrier_destroy(&data_->barrier);
        destroy_result |= pthread_mutex_destroy(&data_->value_mutex);
        destroy_result |= pthread_mutex_destroy(&data_->data_buffer_mutex);
        destroy_result |= pthread_cond_destroy(&data_->value_cond);
        destroy_result |= sem_destroy(&data_->generic_semaphore);
        destroy_result |= pthread_mutex_destroy(&data_->generic_mutex);
            
        if (destroy_result != 0)
        {
            // Log the error but don't throw since this is cleanup
            fprintf(stderr, "Warning: Some pthread objects failed to destroy: %s\n",
                    std::strerror(destroy_result));
        }
        // Unlink the shared memory segment from the system.
        if (shm_unlink(shm_name.c_str()) == -1)
        {
            fprintf(stderr, "Warning: shm_unlink failed: %s\n", std::strerror(errno));
        }
    }

private:
    std::string name_;
    SharedPayload* data_;
    size_t total_size_;
};

// --- pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "A C++ PyTorch extension for creating and sharing CUDA/HIP "
              "tensors using IPC";
    m.def("create_tensor_and_get_ipc", &create_tensor_and_get_ipc,
          "Create a new tensor on CUDA/HIP using a custom IPC-aware "
          "allocator and return its tensor and IPC handle.");
    m.def("copy_tensor_and_get_ipc", &copy_tensor_and_get_ipc,
          "Copies a tensor to new IPC-aware shared memory and returns "
          "the new tensor and its IPC handle.");
    m.def("open_ipc_and_get_tensor", &open_ipc_and_get_tensor,
          "Open a CUDA/HIP IPC handle and returns a tensor.");
    m.def("acquire", &acquire_lock_for_tensor, 
          "Acquire the lock for a given IPC tensor.");
    m.def("release", &release_lock_for_tensor, 
          "Release the lock for a given IPC tensor, includes device-wide "
          "synchronization to avoid race conditions on kernel launched in "
          "the critical section.");
    m.def("release_async", &release_lock_for_tensor_async, 
          "Release the lock for a given IPC tensor, Pytorch-level GPU "
          "synchronization is required to avoid race conditions on kernel "
          "launched in the critical section.");

    py::class_<SharedData>(m, "SharedData")
        .def(py::init<const std::string&, bool, unsigned int, int64_t, 
                      unsigned int, size_t>(), 
            "Creates or opens a shared data object.",
            py::arg("name"), py::arg("is_creator"), 
            py::arg("barrier_size"), py::arg("initial_value") = 0,
            py::arg("semaphore_count") = 1, 
            py::arg("data_buffer_size") = 4*1024)
        .def("get_name", &SharedData::get_name, 
             "Returns the name of the shared data object.")
        .def("add", &SharedData::add, 
             "Atomically adds a value and returns the new value.")
        .def("get", &SharedData::get, 
             "Atomically gets the current value.")
        .def("set", &SharedData::set, 
             "Atomically sets the value.")
        .def("wait_for_value", &SharedData::wait_for_value, 
             "Waits until the shared integer reaches a target value.",
             py::arg("target_value"))

        .def("sem_wait", &SharedData::sem_wait, 
             "Waits on (decrements) the semaphore.")
        .def("sem_post", &SharedData::sem_post, 
             "Posts to (increments) the semaphore.")
        
        .def("mutex_acquire", &SharedData::mutex_acquire, 
             "Acquires the exclusive mutex lock.")
        .def("mutex_release", &SharedData::mutex_release, 
             "Releases the exclusive mutex lock.")
            
        .def("barrier", &SharedData::barrier, 
             "A fast, process-shared barrier.")
        .def("write_data", [](SharedData &self, py::bytes data)
        {
            std::string s = data;
            self.write_data(std::vector<char>(s.begin(), s.end()));
        }, "Writes bytes into the shared buffer.")
        .def("read_data", [](SharedData &self)
        {
            auto data_vec = self.read_data();
            return py::bytes(data_vec.data(), data_vec.size());
        }, "Reads bytes from the shared buffer.")
        .def("destroy", &SharedData::destroy, 
             "Destroys and unlinks the shared memory and semaphore for a "
             "SharedData object.");
}
