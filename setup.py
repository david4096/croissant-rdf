from setuptools import setup, find_packages

setup(
    name='huggingface-rdf',
    version='0.1.0',
    author='David Steinberg',
    author_email='david@resium.com',  # Replace with your email
    description='A command-line tool for generating UMAP plots and KMeans clustering from JSONL data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/david4096/huggingface-rdf',
    packages=find_packages(),
    install_requires=[
        'huggingface_hub',
        'rdflib',
        'qlever',
        'kaggle'
    ],
    entry_points={
        'console_scripts': [
            'huggingface-rdf=huggingface_rdf:main',
            'kaggle-rdf=kaggle_rdf:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)