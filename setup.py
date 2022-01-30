import setuptools

setuptools.setup(
    name="NESDE",
    version="0.0.1",
    author="Anonymous",
    author_email="author@example.com",
    description="Neural Eigen SDE",
    long_description="Implementatin of NESDE algorithm",
    long_description_content_type="text/markdown",
    url="https://github.com/NESDE/NESDE",
    packages=setuptools.find_packages(),
    install_requires = ["numpy","pandas","torch","argparse","tqdm","matplotlib","sdeint","gym"])
