#!/bin/bash
# Based on a script from scikit-learn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.

if [[ "$DISTRIB" == "conda_min" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    export PATH=/home/travis/miniconda/bin:$PATH
    conda config --set always_yes yes
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage \
        six=$SIX_VERSION numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv
    pip install matplotlib

elif [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    export PATH=/home/travis/miniconda/bin:$PATH
    conda config --set always_yes yes
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage six=$SIX_VERSION \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv
    
    if [[ "$COVERAGE" == "true" ]]; then
        pip install coveralls
    fi

elif [[ "$DISTRIB" == "conda_extra" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    export PATH=/home/travis/miniconda/bin:$PATH
    conda config --set always_yes yes
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose coverage six=$SIX_VERSION \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION
    source activate testenv

    pip install scikit-monaco
    pip install matplotlib

    if [[ "$COVERAGE" == "true" ]]; then
        pip install coveralls
    fi


elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # deactivate
    # Create a new virtualenv using system site packages for numpy and scipy
    # virtualenv --system-site-packages testenv
    # source testenv/bin/activate
    pip install nose
    pip install coverage
    pip install numpy==$NUMPY_VERSION
    pip install scipy==$SCIPY_VERSION
    pip install six==$SIX_VERSION
    pip install matplotlib

elif [[ "$DISTRIB" == "ubuntu_extra" ]]; then
    # deactivate
    # Create a new virtualenv using system site packages for numpy and scipy
    # virtualenv --system-site-packages testenv
    # source testenv/bin/activate
    pip install nose
    pip install coverage
    pip install numpy==$NUMPY_VERSION
    pip install scipy==$SCIPY_VERSION
    pip install six==$SIX_VERSION
    pip install scikit-monaco==$SKMONACO_VERSION
    pip install matplotlib==$MATPLOTLIB_VERSION
 
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coveralls
fi

pip install .   # Installs kcsd-python 
python -c "import numpy; import os; assert os.getenv('NUMPY_VERSION') == numpy.__version__"
python -c "import scipy; import os; assert os.getenv('SCIPY_VERSION') == scipy.__version__"
python -c "import kcsd; assert kcsd.__version__ == 1.2"
