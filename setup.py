import os
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

LIBRARY_NAME = "rlaopt"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def find_sources(root_dir, file_ext):
    """Find all files with the given extension recursively starting from root_dir."""
    pattern = os.path.join(root_dir, f"**/*{file_ext}")
    return glob.glob(pattern, recursive=True)


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    use_openmp = os.getenv("USE_OPENMP", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    # Get PyTorch library path for RPATH
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

    extra_link_args = [
        # Add RPATH to ensure PyTorch libraries can be found at runtime
        f"-Wl,-rpath,{torch_lib_path}"
    ]

    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
        ],
    }

    if use_openmp:
        extra_compile_args["cxx"].append("-fopenmp")
        extra_link_args.append("-fopenmp")

    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, LIBRARY_NAME, "csrc")
    extensions_include_dirs = [os.path.join(extensions_dir, "cpp_include")]

    sources = find_sources(extensions_dir, ".cpp")
    if use_cuda:
        sources += find_sources(extensions_dir, ".cu")
        extensions_include_dirs.append(os.path.join(extensions_dir, "cuda_include"))

    ext_modules = [
        extension(
            f"{LIBRARY_NAME}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            # Tells compiler where to find the header files
            include_dirs=extensions_include_dirs,
            library_dirs=[torch_lib_path],  # Add PyTorch library directory
            runtime_library_dirs=[torch_lib_path],  # Add runtime path (RPATH)
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
