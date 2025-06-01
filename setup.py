"""Setup script for multi-camera classification framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multi_camera_classification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Game-theoretic multi-camera classification with energy-dependent accuracy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi_camera_classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "networkx>=2.6.0",
        "torch>=1.9.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "viz": [
            "plotly>=5.3.0",
            "seaborn>=0.11.0",
        ],
    },
)
