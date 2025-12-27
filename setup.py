from setuptools import setup, find_packages

setup(
    name="matar-mini-nn",             
    version="0.1.0",
    author="Tayseer Matar",
    author_email="eng.tayaeermatar@gmail.com",
    description="A simple neural network library from scratch",
    long_description=open("README").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),     
    python_requires=">=3.8",
    install_requires=[
        "numpy",
            ],
    classifiers=[
"Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
