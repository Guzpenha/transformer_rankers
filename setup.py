import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformer_rankers", # Replace with your own username
    version="0.0.1",
    author="Gustavo Penha",
    author_email="guzpenha10@gmail.com",
    description="A library to conduct ranking experiments with transformers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Guzpenha/transformer_rankers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
