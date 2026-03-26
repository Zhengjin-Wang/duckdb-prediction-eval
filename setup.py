from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="duckdb-lcm-eval",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="DuckDB-focused evaluation framework for Learned Cost Models on IMDB dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/duckdb-lcm-eval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "duckdb-lcm-eval=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["datasets/**/*"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/duckdb-lcm-eval/issues",
        "Source": "https://github.com/yourusername/duckdb-lcm-eval",
    },
)