import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="euclid_morphology",
    version="0.0.1",
    author="Mike Walmsley",
    author_email="walmsleymk1@gmail.com",
    description="Morphology predictions by Zoobot for Euclid OU-MER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwalmsley/zoobot-euclid",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7"
)
