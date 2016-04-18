"""
This script is used to generate basis sources for the 
kCSD method Jan et.al (2012) for 3D case.

These scripts are based on Grzegorz Parka's, 
Google Summer of Code 2014, INFC/pykCSD  

This was written by :
Michal Czerwinski, Chaitanya Chintaluri  
Laboratory of Neuroinformatics,
Nencki Institute of Experimental Biology, Warsaw.
"""
import numpy as np

def step_1D(xp, mu, R):
    """Returns normalized 1D step function.

    Parameters
    ----------
    xp : floats or np.arrays
        point or set of points where function should be calculated
    mu : float
        origin of the function
    R : float
        cutoff range

    Returns
    -------
    s : Value of the function (xp-mu[0])**2  <= R**2) / np.pi*R**2
    """
    s = ((xp-mu[0])**2  <= R**2)
    s = s / (np.pi*R**2)
    return s        

def gauss_1D(x, mu, three_stdev):
    """Returns normalized gaussian 2D scale function

    Parameters
    ----------
    x : floats or np.arrays
        coordinates of a point/points at which we calculate the density
    mu : list
        distribution mean vector
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : (three_std/3)*(1/2*pi)*(exp(-0.5)*stddev**(-2) *((x-mu)**2))
    """
    h = 1./(2*np.pi)
    stdev = three_stdev/3.0
    h_n = h*stdev
    Z = h_n*np.exp(-0.5 * stdev**(-2) * ((x - mu)**2 ))
    return Z

def gauss_lim_1D(x, y, mu, three_stdev):
    """Returns gausian 2D function cut off after 3 standard deviations.

    Parameters
    ----------
    x : floats or np.arrays
        coordinates of a point/points at which we calculate the density
    mu : list
        distribution mean vector
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : (three_std/3)*(1/2*pi)*(exp(-0.5)*stddev**(-2) *((x-mu)**2)), 
        cut off = three_stdev
    """
    Z = gauss_2D(x, y, mu, three_stdev)
    Z *= ((x - mu[0])**2 < three_stdev**2)
    return Z

def step_2D(xp, yp, mu, R):
    """Returns normalized 2D step function.

    Parameters
    ----------
    xp, yp : floats or np.arrays
        point or set of points where function should be calculated
    mu : float
        origin of the function
    R : float
        cutoff range
    
    Returns
    -------
    s : step function
    """
    s = ((xp-mu[0])**2 + (yp-mu[1])**2 <= R**2)
    s = s / (np.pi*R**2)
    return s        

def gauss_2D(x, y, mu, three_stdev):
    """Returns normalized gaussian 2D scale function

    Parameters
    ----------
    x, y : floats or np.arrays
        coordinates of a point/points at which we calculate the density
    mu : list
        distribution mean vector
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : function
        Normalized gaussian 2D function
    """
    h = 1./(2*np.pi)
    stdev = three_stdev/3.0
    h_n = h * stdev
    Z = h_n * np.exp(-0.5 * stdev**(-2) * ((x - mu[0])**2 + (y - mu[1])**2))
    return Z

def gauss_lim_2D(x, y, mu, three_stdev):
    """Returns gausian 2D function cut off after 3 standard deviations.

    Parameters
    ----------
    x, y : floats or np.arrays
        coordinates of a point/points at which we calculate the density
    mu : list
        distribution mean vector
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : function
        Normalized gaussian 2D function cut off after three_stdev
    """
    Z = gauss_2D(x, y, mu, three_stdev)
    Z *= ((x - mu[0])**2 + (y - mu[1])**2 < three_stdev**2)
    return Z

def gauss_3D(x, y, z, mu, three_stdev):
    """Returns normalized gaussian 3D scale function

    Parameters
    ----------
    x, y, z : floats or np.arrays
        coordinates of a point/points at which we calculate the density
    mu : list
        distribution mean vector
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : funtion
        Normalized gaussian 3D function
    """
    stdev = three_stdev/3.0
    h = 1./(((2*np.pi)**0.5) * stdev)**3
    c = 0.5 * stdev**(-2)
    Z = h * np.exp(-c * ((x - mu[0])**2 + (y - mu[1])**2 + (z - mu[2])**2))
    return Z

def gauss_lim_3D(x, y, z, mu, three_stdev):
    """Returns normalized gaussian 3D scale function cut off after 3stdev

    Parameters
    ----------
    x, y, z : floats or np.arrays
        coordinates of a point/points at which we calculate the density
    mu : list
        distribution mean vector
    three_stdev : float
        3 * standard deviation of the distribution

    Returns
    -------
    Z : funtion
        Normalized gaussian 3D function cutoff three_Stdev
    """
    Z = gauss_3D(x, y, z, mu, three_stdev)
    Z = Z * ((x - mu[0])**2 + (y - mu[1])**2 + (z - mu[2])**2 < three_stdev**2)
    return Z

def step_3D(xp, yp, zp, mu, R):
    """Returns normalized 3D step function.

    Parameters
    ----------
    xp, yp, zp : floats or np.arrays
        point or set of points where function should be calculated
    mu : float
        origin of the function
    R : float
        cutoff range

    Returns
    -------
    s : step function in 3D
    """
    s = 3/(4*np.pi*R**3)*(xp**2 + yp**2 + zp**2 <= R**2)
    return s

basis_1D = {
    "step": step_1D,
    "gauss": gauss_1D,
    "gauss_lim": gauss_lim_1D,
}


basis_2D = {
    "step": step_2D,
    "gauss": gauss_2D,
    "gauss_lim": gauss_lim_2D,
}

basis_3D = {
    "step": step_3D,
    "gauss": gauss_3D,
    "gauss_lim": gauss_lim_3D,
}
