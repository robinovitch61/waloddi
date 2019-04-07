from setuptools import setup, find_packages

requirements = [
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
]

setup(
    name='waloddi',
    python_requires='>=3.5',
    version='0.0.1',
    description="weibull analysis toolbox",
    author="Leo Robinovitch",
    author_email='leorobinovitch@gmail.com',
    url='https://github.com/robinovitch61/waloddi',
    install_requires=requirements,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=True,
    keywords=find_packages(),
    classifiers=[
        'Intended Audience :: Users',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
