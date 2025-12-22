"""
Setup script for AutoSLM package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autoslm",
    version="1.0.0",
    author="AutoSLM Team",
    description="Agentic SLM Training System - Production Grade",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.37.0",
        "datasets>=2.16.0",
        "accelerate>=0.26.0",
        "peft>=0.8.0",
        "bitsandbytes>=0.42.0",
        "trl>=0.7.10",
        "langgraph>=0.0.20",
        "langchain>=0.1.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.5.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "pyyaml>=6.0.1",
        "rich>=13.7.0",
        "click>=8.1.0",
        "ollama>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.23.0",
        ],
        "mlflow": [
            "mlflow>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autoslm=cli:cli",
        ],
    },
)
