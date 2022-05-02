from setuptools import setup

setup(
    name='gahyparopt',
    version='0.0.1',    
    description='Genetic Algorithm for hyperparameter optimization',
    url='https://github.com/mpievolbio-scicomp/gahyparopt',
    author='Carsten Fortmann-Grote (2020-2022), OpenAI (-2020)',
    author_email='grotec@evolbio.mpg.de',
    license='GPL3',
    packages=['gahyparopt'],
    install_requires=[
        'pandas',
        'ipywidgets',
        'numpy',
        'tensorflow>=2.2',
        'matplotlib',
        'pyocclient'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
