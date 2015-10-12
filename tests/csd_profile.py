from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def csd_profile_large(x,y,z=0):
    '''Same as 'large source' profile in 2012 paper'''
    zz = [0.4, -0.3, -0.1, 0.6] 
    zs = [0.2, 0.3, 0.4, 0.2] 
    f1 = 0.5965*exp( (-1*(x-0.1350)**2 - (y-0.8628)**2) /0.4464)* exp(-(z-zz[0])**2 / zs[0]) /exp(-(zz[0])**2/zs[0])
    f2 = -0.9269*exp( (-2*(x-0.1848)**2 - (y-0.0897)**2) /0.2046)* exp(-(z-zz[1])**2 / zs[1]) /exp(-(zz[1])**2/zs[1]);
    f3 = 0.5910*exp( (-3*(x-1.3189)**2 - (y-0.3522)**2) /0.2129)* exp(-(z-zz[2])**2 / zs[2]) /exp(-(zz[2])**2/zs[2]);
    f4 = -0.1963*exp( (-4*(x-1.3386)**2 - (y-0.5297)**2) /0.2507)* exp(-(z-zz[3])**2 / zs[3]) /exp(-(zz[3])**2/zs[3]);
    f = f1+f2+f3+f4
    return f

def csd_profile_large_rand(x,y,z=0,states=0):
    '''random source based on 'large source' profile in 2012 paper'''
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

def csd_profile_small(x,y,z=0):
    def gauss2d(x,y,p):
        """
         p:     list of parameters of the Gauss-function
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

        g = p[4]*np.exp(-(((rcen_x-xp)/p[2])**2+
                          ((rcen_y-yp)/p[3])**2)/2.)
        return g
    f1 = gauss2d(x,y,[0.3,0.7,0.038,0.058,0.5,0.])
    f2 = gauss2d(x,y,[0.3,0.6,0.038,0.058,-0.5,0.])
    f3 = gauss2d(x,y,[0.45,0.7,0.038,0.058,0.5,0.])
    f4 = gauss2d(x,y,[0.45,0.6,0.038,0.058,-0.5,0.])
    f = f1+f2+f3+f4
    return f

def csd_profile_small_rand(x,y,z=0,states=None):
    def gauss2d(x,y,p):
        """
         p:     list of parameters of the Gauss-function
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

        g = p[4]*np.exp(-(((rcen_x-xp)/p[2])**2+
                          ((rcen_y-yp)/p[3])**2)/2.)
        return g
    angle = states[18]*180.
    x_amp = 0.038
    y_amp = 0.056
    f1 = gauss2d(x,y,[states[12],states[14],x_amp,y_amp,0.5,angle])
    f2 = gauss2d(x,y,[states[12],states[15],x_amp,y_amp,-0.5,angle])
    f3 = gauss2d(x,y,[states[13],states[14],x_amp,y_amp,0.5,angle])
    f4 = gauss2d(x,y,[states[13],states[15],x_amp,y_amp,-0.5,angle])
    f = f1+f2+f3+f4
    return f

if __name__=='__main__':
    csd_profile = csd_profile_large_rand
    #csd_profile = csd_profile_small_rand
    rstate = np.random.RandomState(0) #seed here!
    states = rstate.random_sample(24)
    states[0:12] = 2*states[0:12] -1.

    chrg_x, chrg_y = np.mgrid[0.:1.:50j, 
                              0.:1.:50j]
    f = csd_profile(chrg_x, chrg_y, np.zeros(len(chrg_y)), states=states) 

    fig = plt.figure(1)
    ax1 = plt.subplot(111, aspect='equal')
    im = ax1.contourf(chrg_x, chrg_y, f, 15, cmap=cm.bwr)
    cbar = plt.colorbar(im, shrink=0.5)
    plt.show()
