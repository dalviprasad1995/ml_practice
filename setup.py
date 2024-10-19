from setuptools import setup, find_packages

setup(
    name="Iris logistic regression",
    version="v0.1",
    description="A sample Python package",
    author="Prasad Dalvi",
    author_email="dalviprasad1995@gmail.com",
    url="https://github.com/dalviprasad1995/ml_practice",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "pickle"],
)
