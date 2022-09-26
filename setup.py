from distutils.core import setup
import os

with open("README.rst", "r") as f:
    long_description = f.read()

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DIR_PATH, 'requirements.txt'),
          'r', encoding='utf-8') as f:
    requirements = f.readlines()

setup(name='mengzi-zero-shot',  # 包名
      version='0.2.1',  # 版本号
      description='NLU & NLG (zero-shot) depend on mengzi-t5-base-mt',
      long_description=long_description,
      author='huajingyun',
      author_email='huajingyun@hotmail.com',
      url='https://github.com/Langboat/mengzi-zero-shot',
      install_requires=requirements,
      license='Apache License 2.0',
      packages=['mengzi_zs'],
      platforms=["all"],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='machine-learning,zero-shot,NLU,NLG,NLP',
      )
