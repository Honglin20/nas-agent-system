from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nas-cli",
    version="1.0.0",
    author="NAS Agent Team",
    description="智能 NAS 寻优空间注入 CLI 工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Honglin20/nas-agent-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "libcst>=1.0.0",
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "prompt-toolkit>=3.0.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nas-cli=nas_cli.main:main",
        ],
    },
)
