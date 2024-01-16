import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_file = os.path.dirname(__file__)

setup(
    name="sparse_mask",
    ext_modules=[
        CUDAExtension(
            "mask_gen_cuda",
            [
                "mask_gen/src/mask_gen.cpp",
                "mask_gen/src/mask_gen_kernal.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
