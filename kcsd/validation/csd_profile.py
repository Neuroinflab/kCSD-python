'''
This script is used to generate dummy CSD sources,
to test the various kCSD methods

This script is in alpha phase.

This was written by :
Michal Czerwinski, Chaitanya Chintaluri,
Laboratory of Neuroinformatics,
Nencki Institute of Exprimental Biology, Warsaw.
'''
import numpy as np
from numpy import exp, isfinite
from functools import wraps


def repeatUntilValid(f):
    """
    A decorator (wrapper).

    If output of `f(..., seed)` contains either NaN or infinite, repeats
    calculations for other `seed` (randomly generated for the current `seed`)
    until the result is valid.

    :param f: function of two arguments (the latter is `seed`)
    :return: wrapped function f
    """
    @wraps(f)
    def wrapper(arg, seed=0):
        while True:
            result = f(arg, seed)
            if isfinite(result).all():
                return result

            rstate = np.random.RandomState(seed)
            seed = rstate.randint(2**32 - 1)

    # Python 2.7 walkarround necessary for test purposes
    if not hasattr(wrapper, '__wrapped__'):
        setattr(wrapper, '__wrapped__', f)

    return wrapper


def get_states_1D(seed, n=1):
    """
    Used in the random seed generation
    creates a matrix that will generate seeds, here for gaussians:
    amplitude (-1., 1.), location (0,1)*ndim, sigma(0,1)
    """
    ndim = 1
    if seed == 0:
        states = np.array([1., 0.5, 0.5], ndmin=2)
    rstate = np.random.RandomState(seed)
    states = rstate.random_sample(n * (ndim + 2)).reshape((n, (ndim + 2)))
    states[:, 0] = (2 * states[:, 0]) - 1.
    return states, rstate


def add_1d_gaussians(x, states):
    '''Function used for adding multiple 1D gaussians'''
    f = np.zeros(x.shape)
    for i in range(states.shape[0]):
        gauss = states[i, 0]*np.exp(-((x - states[i, 1])**2)/(2.*states[i, 2])
                                    )*(2*np.pi*states[i, 2])**-0.5
        f += gauss
    return f


def gauss_1d_mono(x, seed=0):
    '''Random monopole in 1D'''
    states, rstate = get_states_1D(seed, n=1)
    f = add_1d_gaussians(x, states)
    return f


def gauss_1d_dipole(x, seed=0):
    '''Random dipole source in 1D'''
    states, rstate = get_states_1D(seed, n=1)
    offset = rstate.random_sample(1) - 0.5
    states = np.tile(states, (2, 1))
    states[1, 0] *= -1.  # A Sink
    states[1, 1] += offset
    f = add_1d_gaussians(x, states)
    return f


def get_states_2D(seed):
    """
    Used in the random seed generation for 2d sources
    """
    rstate = np.random.RandomState(seed)
    states = rstate.random_sample(24)
    states[0:12] = 2*states[0:12] - 1.
    return states


@repeatUntilValid
def gauss_2d_large(csd_at, seed=0):
    '''random quadpolar'large source' profile in 2012 paper in 2D'''
    x, y = csd_at
    states = get_states_2D(seed)
    z = 0
    zz = states[0:4]
    zs = states[4:8]
    mag = states[8:12]
    loc = states[12:20]
    scl = states[20:24]
    f1 = mag[0]*exp( (-1*(x-loc[0])**2 - (y-loc[4])**2) /scl[0])* exp(-(z-zz[0])**2 / zs[0]) /exp(-(zz[0])**2/zs[0])
    f2 = mag[1]*exp( (-2*(x-loc[1])**2 - (y-loc[5])**2) /scl[1])* exp(-(z-zz[1])**2 / zs[1]) /exp(-(zz[1])**2/zs[1]);
    f3 = mag[2]*exp( (-3*(x-loc[2])**2 - (y-loc[6])**2) /scl[2])* exp(-(z-zz[2])**2 / zs[2]) /exp(-(zz[2])**2/zs[2]);
    f4 = mag[3]*exp( (-4*(x-loc[3])**2 - (y-loc[7])**2) /scl[3])* exp(-(z-zz[3])**2 / zs[3]) /exp(-(zz[3])**2/zs[3]);
    f = f1+f2+f3+f4
    return f


def gauss_2d_small(csd_at, seed=0):
    '''random quadpolar small source in 2D'''
    x, y = csd_at

    def gauss2d(x, y, p):
        """
         p:    list of parameters of the Gauss-function
               [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE]
               SIGMA = FWHM / (2*sqrt(2*log(2)))
               ANGLE = rotation of the X,Y direction of the Gaussian in radians
        Returns
        -------
        the value of the Gaussian described by the parameters p
        at position (x,y)
        """
        rcen_x = p[0] * np.cos(p[5]) - p[1] * np.sin(p[5])
        rcen_y = p[0] * np.sin(p[5]) + p[1] * np.cos(p[5])
        xp = x * np.cos(p[5]) - y * np.sin(p[5])
        yp = x * np.sin(p[5]) + y * np.cos(p[5])
        g = p[4]*np.exp(-(((rcen_x-xp)/p[2])**2 +
                          ((rcen_y-yp)/p[3])**2)/2.)
        return g
    states = get_states_2D(seed)
    angle = states[18]*180.
    x_amp = 0.038
    y_amp = 0.056
    f1 = gauss2d(x, y, [states[12], states[14], x_amp, y_amp, 0.5, angle])
    f2 = gauss2d(x, y, [states[12], states[15], x_amp, y_amp, -0.5, angle])
    f3 = gauss2d(x, y, [states[13], states[14], x_amp, y_amp, 0.5, angle])
    f4 = gauss2d(x, y, [states[13], states[15], x_amp, y_amp, -0.5, angle])
    f = f1+f2+f3+f4
    return f


def get_states_3D(seed):
    """
    Used in the random seed generation for 3D sources
    """
    rstate = np.random.RandomState(seed)  # seed here!
    states = rstate.random_sample(24)
    return states


def gauss_3d_small(csd_at, seed=0):
    '''A random quadpole small souce in 3D'''
    x, y, z = csd_at
    states = get_states_3D(seed)
    x0, y0, z0 = states[0:3]
    x1, y1, z1 = states[3:6]
    if states[6] < 0.01:
        states[6] *= 25
    sig_2 = states[6] / 75.
    p1, p2, p3 = (ii*0.5 for ii in states[8:11])
    A = (2*np.pi*sig_2)**-1
    f1 = A*np.exp((-(x-x0)**2 - (y-y0)**2 - (z-z0)**2) / (2*sig_2))
    f2 = -1*A*np.exp((-(x-x1)**2 - (y-y1)**2 - (z-z1)**2) / (2*sig_2))
    x2 = np.modf(x0+p1)[0]
    y2 = np.modf(y0+p2)[0]
    z2 = np.modf(z0+p3)[0]
    f3 = A*np.exp((-(x-x2)**2 - (y-y2)**2 - (z-z2)**2) / (2*sig_2))
    x3 = np.modf(x1+p1)[0]
    y3 = np.modf(y1+p2)[0]
    z3 = np.modf(z1+p3)[0]
    f4 = -1*A*np.exp((-(x-x3)**2 - (y-y3)**2 - (z-z3)**2) / (2*sig_2))
    f = f1+f2+f3+f4
    return f


def gauss_3d_large(csd_at, seed=0):
    '''A random dipolar Large source in 3D'''
    x, y, z = csd_at
    states = get_states_3D(seed)
    x0, y0, z0 = states[7:10]
    x1, y1, z1 = states[10:13]
    if states[1] < 0.01:
        states[1] *= 25
    sig_2 = states[1] * 5
    A = (2*np.pi*sig_2)**-1
    f1 = A*np.exp((-(x-x0)**2 - (y-y0)**2 - (z-z0)**2) / (2*sig_2))
    f2 = -1*A*np.exp((-(x-x1)**2 - (y-y1)**2 - (z-z1)**2) / (2*sig_2))
    f = f1+f2
    return f


def gauss_1d_dipole_f(x):
    """1D Gaussian dipole source is placed between 0 and 1
       to be used to test the CSD

       Parameters
       ----------
       x : np.array
           Spatial pts. at which the true csd is evaluated

       Returns
       -------
       f : np.array
           The value of the csd at the requested points
    """
    src = 0.5*exp(-((x-0.7)**2)/(2.*0.3))*(2*np.pi*0.3)**-0.5
    snk = -0.5*exp(-((x-0.3)**2)/(2.*0.3))*(2*np.pi*0.3)**-0.5
    f = src+snk
    return f


def gauss_2d_small_f(csd_at):
    '''Source from Jan 2012 kCSD  paper'''
    x, y = csd_at

    def gauss2d(x, y, p):
        """
         p:    list of parameters of the Gauss-function
               [XCEN,YCEN,SIGMAX,SIGMAY,AMP,ANGLE]
               SIGMA = FWHM / (2*sqrt(2*log(2)))
               ANGLE = rotation of the X,Y direction of the Gaussian in radians
        Returns
        -------
        the value of the Gaussian described by the parameters p
        at position (x,y)
        """
        rcen_x = p[0] * np.cos(p[5]) - p[1] * np.sin(p[5])
        rcen_y = p[0] * np.sin(p[5]) + p[1] * np.cos(p[5])
        xp = x * np.cos(p[5]) - y * np.sin(p[5])
        yp = x * np.sin(p[5]) + y * np.cos(p[5])
        g = p[4]*np.exp(-(((rcen_x-xp)/p[2])**2 +
                          ((rcen_y-yp)/p[3])**2)/2.)
        return g
    f1 = gauss2d(x, y, [0.3, 0.7, 0.038, 0.058, 0.5, 0.])
    f2 = gauss2d(x, y, [0.3, 0.6, 0.038, 0.058, -0.5, 0.])
    f3 = gauss2d(x, y, [0.45, 0.7, 0.038, 0.058, 0.5, 0.])
    f4 = gauss2d(x, y, [0.45, 0.6, 0.038, 0.058, -0.5, 0.])
    f = f1+f2+f3+f4
    return f


def gauss_2d_large_f(csd_at):
    '''Fixed 'large source' profile in 2012 paper'''
    x, y = csd_at
    z = 0
    zz = [0.4, -0.3, -0.1, 0.6]
    zs = [0.2, 0.3, 0.4, 0.2]
    f1 = 0.5965*exp((-1*(x-0.1350)**2 - (y-0.8628)**2) / 0.4464)*exp(-(z-zz[0])**2 / zs[0]) / exp(-(zz[0])**2/zs[0])
    f2 = -0.9269*exp((-2*(x-0.1848)**2 - (y-0.0897)**2) / 0.2046)*exp(-(z-zz[1])**2 / zs[1]) / exp(-(zz[1])**2/zs[1])
    f3 = 0.5910*exp((-3*(x-1.3189)**2 - (y-0.3522)**2) / 0.2129)*exp(-(z-zz[2])**2 / zs[2]) / exp(-(zz[2])**2/zs[2])
    f4 = -0.1963*exp((-4*(x-1.3386)**2 - (y-0.5297)**2) / 0.2507)*exp(-(z-zz[3])**2 / zs[3]) / exp(-(zz[3])**2/zs[3])
    f = f1+f2+f3+f4
    return f


def gauss_3d_dipole_f(csd_at):
    '''Fixed dipole in 3 dimensions of the volume'''
    x, y, z = csd_at
    x0, y0, z0 = 0.3, 0.7, 0.3
    x1, y1, z1 = 0.6, 0.5, 0.7
    sig_2 = 0.023
    A = (2*np.pi*sig_2)**-1
    f1 = A*np.exp((-(x-x0)**2 - (y-y0)**2 - (z-z0)**2) / (2*sig_2))
    f2 = -1*A*np.exp((-(x-x1)**2 - (y-y1)**2 - (z-z1)**2) / (2*sig_2))
    f = f1+f2
    return f


def gauss_3d_mono1_f(csd_at):
    '''Fixed monopole in 3D at the center of the volume space'''
    x, y, z = csd_at
    x0, y0, z0 = 0.5, 0.5, 0.5
    sig_2 = 0.023
    A = (2*np.pi*sig_2)**-1
    f1 = A*np.exp((-(x-x0)**2 - (y-y0)**2 - (z-z0)**2) / (2*sig_2))
    return f1


def gauss_3d_mono2_f(csd_at):
    '''Fixed monopole in 3D Offcentered wrt volume'''
    x, y, z = csd_at
    x0, y0, z0 = 0.41, 0.41, 0.585
    sig_2 = 0.023
    A = (2*np.pi*sig_2)**-1
    f1 = A*np.exp((-(x-x0)**2 - (y-y0)**2 - (z-z0)**2) / (2*sig_2))
    return f1


def gauss_3d_mono3_f(csd_at):
    '''Fixed monopole in 3D Offcentered wrt volume'''
    x, y, z = csd_at
    x0, y0, z0 = 0.55555556, 0.55555556, 0.55555556
    stdev = 0.3
    h = 1./((2*np.pi)**0.5 * stdev)**3
    c = 0.5*stdev**(-2)
    f1 = h*np.exp(-c*((x - x0)**2 + (y - y0)**2 + (z - z0)**2))
    return f1


# def neat_4d_plot(csd_at, t, z_steps=5, cmap=cm.bwr_r):
#     '''Used to show 3D csd profile'''
#     x, y, z = csd_at
#     t_max = np.max(np.abs(t))
#     levels = np.linspace(-1*t_max, t_max, 15)
#     ind_interest = np.mgrid[0:z.shape[2]:np.complex(0,z_steps+2)]
#     ind_interest = np.array(ind_interest, dtype=np.int)[1:-1]
#     fig = plt.figure(figsize=(4,12))
#     height_ratios = [1 for i in range(z_steps)]
#     height_ratios.append(0.1)
#     gs = gridspec.GridSpec(z_steps+1, 1, height_ratios=height_ratios)
#     for ii, idx in enumerate(ind_interest):
#         ax = plt.subplot(gs[ii, 0])
#         im = plt.contourf(chrg_x[:, :, idx], chrg_y[:, :, idx], t[:, :, idx],
#                           levels=levels, cmap=cmap)
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         title = str(z[:,:,idx][0][0])[:4]
#         ax.set_title(label=title, fontdict={'x':0.8, 'y':0.7})
#         ax.set_aspect('equal')
#     cax = plt.subplot(gs[z_steps,0])
#     cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
#     cbar.set_ticks(levels[::2])
#     cbar.set_ticklabels(np.around(levels[::2], decimals=2))
#     gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
#     #plt.tight_layout()


csd_available_dict = {1: [gauss_1d_mono, gauss_1d_dipole],
                      2: [gauss_2d_large, gauss_2d_small],
                      3: [gauss_3d_large, gauss_3d_small]}


if __name__ == '__main__':
    seed = 3

    # 1D CASE
    csd_profile = gauss_1d_mono
    chrg_x = np.arange(0., 1., 1./50.)
    f = csd_profile(chrg_x, seed)
    # plt.plot(chrg_x, f)

    # #2D CASE
    # csd_profile = gauss_2d_large
    # states[0:12] = 2*states[0:12] -1. #obtain values between -1 and 1
    # chrg_x, chrg_y = np.mgrid[0.:1.:50j,
    #                           0.:1.:50j]
    # f = csd_profile(chrg_x, chrg_y, seed=seed)
    # fig = plt.figure(1)
    # ax1 = plt.subplot(111, aspect='equal')
    # im = ax1.contourf(chrg_x, chrg_y, f, 15, cmap=cm.bwr_r)
    # cbar = plt.colorbar(im, shrink=0.5)
    # plt.show()

    # #3D CASE
    # csd_profile = gauss_3d_small
    # chrg_x, chrg_y, chrg_z = np.mgrid[0.:1.:50j,
    #                                   0.:1.:50j,
    #                                   0.:1.:50j]
    # f = csd_profile(chrg_x, chrg_y, chrg_z, seed=seed)
    # neat_4d_plot(chrg_x, chrg_y, chrg_z, f)

    # plt.show()

    # test of gauss_2d_large(seed=63) -> NaN fix
    csd_at = np.mgrid[0.:1.:100j, 0.:1.:100j]
    for seed in range(63):
        assert (gauss_2d_large.__wrapped__(csd_at, seed) == gauss_2d_large(csd_at, seed)).all(),\
               "decorated gauss_2d_large output differs for seed={}".format(seed)

    assert isfinite(gauss_2d_large(csd_at, 63)).all(),\
           "invalid output of gauss_2d_large(seed=63)"