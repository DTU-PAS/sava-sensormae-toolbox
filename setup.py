import subprocess
import sys

try:
    import setuptools
except ImportError:
    print("setuptools not found, installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req_file:
    install_requires = req_file.read().splitlines()

setuptools.setup(
    name="sava_sensormae_toolbox",
    version="0.1.0",
    author="PAS DTU Electro",
    author_email="fabmo@dtu.dk",
    description="2D/3D Object Detection Inference Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DTU-PAS/sava-sensormae-toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires=">=3.8",
)
