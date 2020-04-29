from distutils.core import setup

setup(
    name="hypertorch",
    version="0.0.3",
    packages=["hypertorch"],
    license="MIT License",
    depends=["requests"],
    description="High level pytorch module programming framework",
    install_requires=["torch","numpy","toolz"]
)