import os
import re
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "rlaopt"


def parse_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_version():
    with open(os.path.join(library_name, "__init__.py"), "r", encoding="utf-8") as f:
        match = re.search(r'__version__ = "(.*?)"', f.read())
        return match.group(1) if match else "0.0.0"


if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    use_openmp = os.getenv("USE_OPENMP", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
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
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    version=get_version(),
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=parse_requirements("requirements.txt"),
    author="Pratik Rathore, Zachary Frangella, Weimu Lei",
    author_email="pratikr@stanford.edu, zfran@stanford.edu, leiweimu@gmail.com",
    description="Randomized linear algebra-based methods for optimization",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/udellgroup/rlaopt",
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/udellgroup/rlaopt",
        "Issue Tracker": "https://github.com/udellgroup/rlaopt/issues",
    },
    python_requires=">=3.9",  # Based on the Py_LIMITED_API setting
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
