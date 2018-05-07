"""
Python implementation of kernel Current Source Density method
"""
from setuptools import setup, find_packages


def readme():
    """
    Used for README
    """
    with open('README.rst') as f:
        return f.read()


setup(name='kcsd',
      version='1.1.2',
      description='kernel current source density methods',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',
          'Intended Audience :: Science/Research',
      ],
      keywords='Eletrophysiology CSD LFP MEA',
      url='https://github.com/Neuroinflab/kCSD-python',
      author='Chaitanya Chintaluri',
      license='BSD',
      packages=find_packages(),
      # package_data={'tests': [
      # https://stackoverflow.com/questions/14422340/manifest-in-package-data-and-data-files-clarification
      # https://github.com/pypa/sampleproject/issues/30#issuecomment-143947944
      #     os.path.join('sKCSD', 'test.mat')
      # ]},
      include_package_data=True,
      install_requires=['future>=0.16.0',
                        'numpy>=1.10.0',
                        'scipy>=0.14.0',
                        'matplotlib>=2.0'],
      extras_require={'skmonaco': ['scikit-monaco>=0.2'],
                      'docs': ['numpydoc>=0.5',
                               'sphinx>=1.2.2']},
      test_suite='kcsd.tests',
      zip_safe=False)
