from setuptools import setup, find_packages

setup(
    name = 'util_pyfmri',
    version = '0.1',
    author = 'Jordan Theriault',
    author_email = 'jordan_theriault@northeastern.edu',
    packages = find_packages(exclude=['examples', 'jt_7t.py']),
    license = 'LICENSE',
    long_description='a personal collection of python tools to simplify analysis using nipype.',
)
