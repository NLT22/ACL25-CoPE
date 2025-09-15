#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from environment.yaml or create basic requirements
def read_requirements():
    requirements = [
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'transformers>=4.20.0',
        'numpy>=1.21.0',
        'pillow>=8.3.0',
        'pyyaml>=5.4.0',
        'tqdm>=4.62.0',
        'tensorboard>=2.7.0',
        'scikit-learn>=1.0.0',
    ]
    return requirements

setup(
    name='acl25_cope',
    version='0.1.0',
    description='ACL25 CoPE: Composed Image Retrieval Models',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='ACL25 CoPE Team',
    author_email='',
    url='https://github.com/ACL25-CoPE',
    packages=find_packages(exclude=['temp', 'temp.*', 'reference', 'reference.*']),
    include_package_data=True,
    package_data={
        'acl25_cope': ['config-*.yaml'],
    },
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'isort>=5.9',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    entry_points={
        'console_scripts': [
            'cope-train=acl25_cope.train:main',
            'cope-eval=acl25_cope.eval:main',
            'cope-list-models=acl25_cope.list_models:main',
        ],
    },
)
