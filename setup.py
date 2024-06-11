import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

with pathlib.Path("requirements_gpu.txt").open() as f:
    extra_gpu = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

with pathlib.Path("requirements_cpu.txt").open() as f:
    extra_cpu = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

setup(
    name="landingai",
    version=0.1,
    author="Sisyphes",
    author_email="fq1123581321@sina.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    description="Simple ai model optimizer and deployment tool",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/foocker/landingai",
    package_dir={"": "landingai"},
    packages=find_packages(where="landingai"),
    install_requires=install_requires,
    extras_require={
        "GPU": extra_gpu,
        "CPU": extra_cpu,
    },
    python_requires=">=3.8.0",
    entry_points={
        "console_scripts": [
            "convert_model = landingai.convert:entrypoint",
        ],
    },
)
