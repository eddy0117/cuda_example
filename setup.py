from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension


sources = ['torch_ext_test.cpp', 'torch_ext_test_cuda.cu']

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
NVCC_FLAGS = ["-O2", "-std=c++17"]

setup(
    name='torch_ext_test',
    ext_modules=[
        CUDAExtension(
            name='torch_ext_test',
            sources=sources,
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
                })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)