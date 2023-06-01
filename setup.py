from setuptools import setup

setup(
    name='podder',
    version='1.0.0',
    packages=['podder'],
    license='MIT',
    author='VonSquiggles',
    description=' ',
    install_requires=[
        'pydub',
        'transformers'
    ],
    dependency_links=[
        'https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip',
        'https://github.com/VonSquiggles/soxan/archive/refs/tags/v0.1.0.tar.gz',
    ]
)