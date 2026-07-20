# -*- coding: utf-8 -*-

from glob import glob
from pathlib import Path

from setuptools import Extension, setup


README = Path('README.md').read_text(encoding='utf-8')

include_dirs = ['src']
libraries = ['speexdsp', 'stdc++']
define_macros = []
extra_compile_args = []

sources = (
    glob('src/echo_canceller.cpp') +
    ['src/speexdsp.i']
)

swig_opts = (
    ['-c++'] +
    ['-I' + h for h in include_dirs]
)


setup(
    name='speexdsp',
    version='0.1.2',
    description='Python bindings of speexdsp library',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Yihui Xiong',
    author_email='yihui.xiong@hotmail.com',
    maintainer='Glucius-TM',
    maintainer_email='yihui.xiong@hotmail.com',
    url='https://github.com/Glucius-TM/speexdsp-python',
    download_url='https://github.com/Glucius-TM/speexdsp-python',
    packages=['speexdsp'],
    package_dir={'speexdsp': 'src'},
    ext_modules=[
        Extension(
            name='speexdsp._speexdsp',
            sources=sources,
            swig_opts=swig_opts,
            include_dirs=include_dirs,
            libraries=libraries,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
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
)
