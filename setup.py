from distutils.core import setup

setup(
    name="hypertorch",
    version="0.0.1",
    packages=["pybatchclient"],
    license="MIT License",
    depends=["requests"],
    description="High level pytorch module programming framework",
    install_requires=["pytorch"]
)