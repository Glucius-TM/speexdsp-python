# -*- coding: utf-8 -*-

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


README = Path('README.md').read_text(encoding='utf-8')

ext_modules = [
    Pybind11Extension(
        name='speexdsp._speexdsp',
        sources=['src/bindings.cpp', 'src/echo_canceller.cpp'],
        include_dirs=['src'],
        libraries=['speexdsp'],
        cxx_std=14,
    )
]


setup(
    name='speexdsp',
    version='0.2.0',
    description='Python bindings of speexdsp library',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Yihui Xiong',
    author_email='yihui.xiong@hotmail.com',
    maintainer='Glucius-TM',
    url='https://github.com/Glucius-TM/speexdsp-python',
    download_url='https://github.com/Glucius-TM/speexdsp-python',
    packages=['speexdsp'],
    package_dir={'speexdsp': 'src'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
    ],
    license='BSD',
    keywords=['speexdsp', 'acoustic echo cancellation'],
    platforms=['Linux'],
    python_requires='>=3.8',
    install_requires=['numpy'],
)
