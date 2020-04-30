from distutils.core import setup

setup(
    name="hypertorch",
    version="0.0.5",
    packages=["hypertorch"],
    license="MIT License",
    depends=["torch","numpy","toolz"],
    description="High level pytorch module programming framework",
    install_requires=[]
)