from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "scipy",
    "sympy",
]

setup(
    author="Drake Wong",
    author_email='drakejwong@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Efficient, interoperable kernel creation and combination.",
    entry_points={
        'console_scripts': [
            'kernelbridge=kernelbridge.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='kernelbridge',
    name='kernelbridge',
    packages=find_packages(exclude="tests"),
    test_suite="tests",
    url='https://github.com/drakejwong/kernelbridge',
    version='0.1a1',
    zip_safe=True,
)
