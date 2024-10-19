from setuptools import setup, find_packages

setup(
    name='iljicevs_ml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'hyperopt'
    ],
    description='Library implementation of the Iljiceva model.',
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
