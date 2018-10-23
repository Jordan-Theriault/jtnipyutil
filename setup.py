from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='jtnipyutil',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Personal utility functions for Nipype",
    author="Jordan Theriault",
    author_email='jtheriault7@gmail.com',
    url='https://github.com/Jordan-Theriault/jtnipyutil',
    packages=['jtnipyutil'],
    entry_points={
        'console_scripts': [
            'jtnipyutil=jtnipyutil.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='jtnipyutil',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ]
)
