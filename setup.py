from setuptools import setup, find_packages

setup(
    name='clustercode',
    version='0.2.0',
    url='https://tbd.com',
    author='Matthias Kiesel, Tom Lindeboom',
    author_email='m.kiesel18@imperial.ac.uk',
    description='Modules designed to evaluate structural properties',
    packages=find_packages(),    
    install_requires=['numpy', 'matplotlib', 'scipy','MDAnalysis => 0.19.2'],
)
