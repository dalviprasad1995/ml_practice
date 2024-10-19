from setuptools import setup, find_packages

setup(
    name="Iris logistic regression",
    version="1.1",
    description="A sample Python package",
    author="Prasad Dalvi",
    author_email="dalviprasad1995@gmail.com",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "pickle"],
)
