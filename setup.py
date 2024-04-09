from setuptools import setup

install_requires = [
    "numpy",
    "pytest",
    "frozendict",
]


setup(
    name="ngrams",
    install_requires=install_requires,
    version="0.1",
    scripts=[],
    packages=["ngrams"],
)
