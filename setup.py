from setuptools import setup, find_packages

setup(
    name="cuda-thermal-denoise-jetson",
    version="1.0.0",
    description="CUDA-Accelerated Image De-noising Pipeline for High-Resolution Industrial Thermal Images on NVIDIA Jetson Orion Nano",
    author="CUDA Thermal Vision Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pillow>=9.0.0",
        "matplotlib>=3.4.0",
        "pytest>=7.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
