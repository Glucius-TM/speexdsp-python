# -*- coding: utf-8 -*-

import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


README = Path('README.md').read_text(encoding='utf-8')


def _split_env_paths(name: str) -> list[str]:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return []
    return [part for part in raw.split(os.pathsep) if part]


include_dirs = ['src'] + _split_env_paths('SPEEXDSP_INCLUDE_DIR')
library_dirs = _split_env_paths('SPEEXDSP_LIBRARY_DIR')

ext_modules = [
    Pybind11Extension(
        name='speexdsp._speexdsp',
        sources=['src/bindings.cpp', 'src/echo_canceller.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
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
    package_data={'speexdsp': ['py.typed']},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
    ],
    license='BSD',
    keywords=['speexdsp', 'acoustic echo cancellation'],
    platforms=['Linux', 'Windows', 'MacOS'],
    python_requires='>=3.8',
    install_requires=['numpy'],
)
