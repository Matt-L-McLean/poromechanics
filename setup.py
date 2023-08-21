import setuptools
  
with open("README.md", "r") as fh:
    description = fh.read()
  
setuptools.setup(
    name="poromechanics",
    version="0.0.1",
    author="Matt_McLean",
    author_email="matthewmclean@utexas.edu",
    packages=["poromechanics"],
    description="Coupled geomechanics simulation",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/Matt-L-McLean/poromechanics",
    install_requires=[]
)