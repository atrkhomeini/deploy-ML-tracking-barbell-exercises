import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "PSO"
AUTHOR_USER_NAME = "atrkhomeini"
SRC_REPO = "deploy-ML-tracking-barbell-exercises"
AUTHOR_EMAIL = "5026211156@student.its.ac.id"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="This is a continuation of the previous ml tracking barbell execises. here the author develops until it can be deployed to the public or can be called MLOps. I wish it'll works",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)