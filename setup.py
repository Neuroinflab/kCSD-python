"""
Python implementation of kernel Current Source Density method
"""
from setuptools import setup, find_packages
import sys


def readme():
    """
    Used for README
    """
    if sys.version_info.major < 3:
        print('No support for python versions < 3.8')
        with open('README.rst') as f:
            return f.read().decode('utf-8')
    else:
        with open('README.rst', encoding='utf-8') as f:
            return f.read()


setup(name='kcsd',
      version='2.0',
      description='kernel current source density methods',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Intended Audience :: Science/Research',
      ],
      keywords='Eletrophysiology CSD LFP MEA',
      url='https://github.com/Neuroinflab/kCSD-python',
      author='Chaitanya Chintaluri et al',
      license='BSD',
      packages=find_packages(where="."),
      include_package_data=True,
      package_data={'kcsd':
                    ['data/*',
                     'data/ball_and_stick_128/*',
                     'data/ball_and_stick_16/*',
                     'data/ball_and_stick_8/*',
                     'data/gang_7x7_200/*',
                     'data/morphology/*',
                     'data/Simple_with_branches/*',
                     'data/Y_shaped_neuron/*',
                     'tutorials/*',
                     'figures/*'
                     ]
                    },
      install_requires=['numpy>=1.8.0',
                        'scipy>=0.14.0',
                        'matplotlib>=2.0'],
      extras_require={'docs': ['numpydoc>=0.5',
                               'sphinx>=1.2.2']},
      test_suite='kcsd.tests',
      zip_safe=False)
