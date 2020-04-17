# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="caption",
    version="1.0.0",
    author="Ricardo Rei, Nuno Miguel Guerreiro",
    download_url="https://gitlab.com/Unbabel/discovery-team/caption",
    author_email="ricardo.rei@unbabel.com, nuno.guerreiro@unbabel.com",
    packages=find_packages(exclude=["tests"]),
    description="Provides automatic transcription enrichment for ASR data",
    keywords=["Deep Learning", "PyTorch", "AI", "NLP", "Natural Language Processing"],
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=[
        line.strip() for line in open("requirements.txt", "r").readlines()
    ],
)
