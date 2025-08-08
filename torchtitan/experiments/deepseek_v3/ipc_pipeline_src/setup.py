import os
import torch
import torch.version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# --- Common Flags and Source Files ---
extra_compile_args = ['-std=c++17', '-g']
libraries = []
include_dirs = []
library_dirs = []

# For POSIX shared memory and semaphores, we need to link against the
# real-time library. This is required by the SharedData class.
libraries.append('rt')

# --- Configure for GPU environment ---
if hasattr(torch.version, 'hip') and torch.version.hip is not None:
    # If we are in a HIP environment, pass a custom macro to the C++ compiler.
    extra_compile_args.append('-DUSE_HIP')
    
    # The ROCM_PATH environment variable is standard on ROCm systems.
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    include_dirs.append(os.path.join(rocm_path, 'include'))
    include_dirs.append(os.path.join(rocm_path, 'roctracer', 'include'))
    library_dirs.append(os.path.join(rocm_path, 'lib'))
    
    # Add the specific library needed for roctx functions
    # libraries.append('roctracer64')
    
else:
    # CUDA environment - add CUDA include and library directories
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    if os.path.exists(cuda_home):
        include_dirs.append(os.path.join(cuda_home, 'include'))
        library_dirs.append(os.path.join(cuda_home, 'lib64'))
        # Add nvtx for profiling if in a CUDA environment.
        # libraries.append('nvtx3')
    else:
        # Fallback: try to find CUDA in common locations
        common_cuda_paths = ['/usr/local/cuda', '/usr/cuda', '/opt/cuda']
        for cuda_path in common_cuda_paths:
            if os.path.exists(cuda_path):
                include_dirs.append(os.path.join(cuda_path, 'include'))
                library_dirs.append(os.path.join(cuda_path, 'lib64'))
                # libraries.append('nvtx3')
                break

# --- Define the single source file to be compiled ---
source_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "torch_ipc_extension.cpp" # The new, combined source file
)

# --- Setup Call ---
setup(
    # The name of the overall package.
    name='torch_ipc_extension',
    ext_modules=[
        # Define a single extension module that compiles the combined C++ file.
        CppExtension(
            # The name of the final importable Python module.
            'torch_ipc_extension', 
            [source_file], # Pass the single source file as a list
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
