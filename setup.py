from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='iljicevs_ml',
    version='0.2.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'hyperopt',
        'imblearn',
        'seaborn',
        'tpot',
        'python-docx',
        'openpyxl',
        'torch',
        'causalml',
        'pandas'
    ],
    description='Library implementation of the Iljiceva model.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Mihails Iljicevs',
    author_email='goenzi61@gmail.com',
    url='https://github.com/ilyacartwright/iljicevs_ml',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
