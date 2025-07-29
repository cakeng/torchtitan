import os
import torch
import torch.version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Define compiler flags and directories.
extra_compile_args = []
include_dirs = []
library_dirs = []

# Check if we are in a HIP environment and configure paths accordingly.
if hasattr(torch.version, 'hip') and torch.version.hip is not None:
    # If we are in a HIP environment, pass a custom macro to the C++ compiler.
    extra_compile_args.append('-DUSE_HIP')
    
    # Add HIP include and library paths.
    # The ROCM_PATH environment variable is standard on ROCm systems.
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    include_dirs.append(os.path.join(rocm_path, 'include'))
    library_dirs.append(os.path.join(rocm_path, 'lib'))

source_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ipc_extension.cpp"
)

setup(
    name='torch_ipc_extension',
    ext_modules=[
        CppExtension(
            'torch_ipc_extension', # The name of the importable module
            [source_file],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)