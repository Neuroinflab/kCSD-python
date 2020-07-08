import kCSD2D_reconstruction_from_npx as npx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import numpy as np
from scipy.signal import filtfilt, butter


def set_axis(ax, x, y, letter=None):
    ax.text(
        x,
        y,
        letter,
        fontsize=20,
        weight='bold',
        transform=ax.transAxes)
    return ax


def make_plot(ax, xx, yy, zz, ele_pos, title='True CSD', cmap=cm.bwr, ylabel=False):
    ax.set_aspect('auto')
    tmax = np.max(abs(zz))
    levels = np.linspace(-tmax, tmax, 251)
    #levels = np.linspace(zz.min(), -zz.min(), 61)
    im = ax.contourf(xx, -(yy-500), zz, levels=levels, cmap=cmap)
    ax.set_xlabel('X ($\mu$m)')
    if ylabel:
        ax.set_ylabel('Y ($\mu$m)')
    ax.set_title(title)
    if cmap=='bwr': 
        plt.colorbar(im, orientation='horizontal',  format='%.2f', ticks=[-0.02,0,0.02])
    else: plt.colorbar(im, orientation='horizontal',  format='%.1f', ticks=[-0.6,0,0.6])
    plt.scatter(ele_pos[:, 0], 
                -(ele_pos[:, 1]-500),
                s=0.8, color='black')
    # plt.gca().invert_yaxis()
    return ax


def fetch_electrodes(meta):
    imroList = meta['imroTbl'].split(sep=')')
    nChan = len(imroList) - 2
    electrode = np.zeros(nChan, dtype=int)
    channel = np.zeros(nChan, dtype=int)
    bank = np.zeros(nChan, dtype=int)
    reference_electrode = np.zeros(nChan, dtype=int)
    for i in range(nChan):
        currList = imroList[i+1].split(sep=' ')
        channel[i] = int(currList[0][1:])
        bank[i] = int(currList[1])
        reference_electrode[i] = currList[2]
    # Channel N => Electrode (1+N+384*A), where N = 0:383, A=0:2
    electrode = 1 + channel + 384 * bank
    return electrode, channel, reference_electrode

def create_electrode_map(start_x, start_y):
    x_dist = 16 #um
    y_dist = 20
    ele_map = {}
    ele_list = []
    for i in range(960):
        x_pos = start_x+(i%2)*x_dist*2+int(((i/2)%2))*x_dist
        y_pos = int(i/2)*y_dist
        ele_map[i+1] = (x_pos, y_pos)
        ele_list.append((i+1, x_pos, y_pos))
    return ele_map, ele_list

def eles_to_coords(eles):
    ele_map, ele_list = create_electrode_map(-24, 0)
    coord_list = []
    for ele in eles:
        coord_list.append(ele_map[ele])
    return np.array(coord_list)

def get_npx(path, time_start, time_stop):
    binFullPath = Path(path)
    meta = readSGLX.readMeta(binFullPath)
    rawData = readSGLX.makeMemMapRaw(binFullPath, meta)
    # get electrode positions
    electrodes, channels, reference_electrode = fetch_electrodes(meta)
    ele_pos_def = eles_to_coords(electrodes)
    # convData is the potential in uV or mV
    Fs = readSGLX.SampRate(meta)
    time_start, time_stop = int(time_start*Fs), int(time_stop*Fs)
    selectData = rawData[:, time_start:time_stop]
    if meta['typeThis'] == 'imec': rawData = 1e3*readSGLX.GainCorrectIM(selectData, channels, meta)
    else: rawData = 1e3*readSGLX.GainCorrectNI(rawData, channels, meta)
    return rawData, ele_pos_def, channels, meta, reference_electrode


def calculate_eigensources(obj):        
    try:
        eigenvalue, eigenvector = np.linalg.eigh(obj.k_pot +
                                                 obj.lambd *
                                                 np.identity
                                                 (obj.k_pot.shape[0]))
        print('lambd: ', obj.lambd)
    except LinAlgError:
        raise LinAlgError('EVD is failing - try moving the electrodes'
                          'slightly')
    idx = eigenvalue.argsort()[::-1]
    eigenvalues = eigenvalue[idx]
    eigenvectors = eigenvector[:, idx]
    eigensources = np.dot(obj.k_interp_cross, eigenvectors)
    return eigensources


def csd_into_eigensource_projection(csd, eigensources):
    orthn = scipy.linalg.orth(eigensources)
    return np.matmul(np.matmul(csd, orthn), orthn.T)


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
    d1source = gauss2d(x, y, [0.0004, 0.34, 0.002, 0.015, 0.018, 0.])
    d1sink = gauss2d(x, y, [0.0015, 0.45, 0.002, 0.015, -0.001, 0.])
    d2sink = gauss2d(x, y, [0.008, 0.34, 0.0025, 0.015, -0.02, 0.])
    d2source = gauss2d(x, y, [0.007, 0.45, 0.0021, 0.02, 0.0075, 0.])
    d3source = gauss2d(x, y, [0.0135, 0.34, 0.0025, 0.015, 0.021, 0.])
    d3sink = gauss2d(x, y, [0.012, 0.45, 0.002, 0.015, -0.001, 0.])
    d4sink = gauss2d(x, y, [0.0205, 0.34, 0.0015, 0.015, -0.02, 0.])
    d4source = gauss2d(x, y, [0.0205, 0.45, 0.0021, 0.02, 0.0075, 0.])
    d5source = gauss2d(x, y, [0.01, 0.51, 0.1, 0.059, 0.009, 0.])
    d5sink = gauss2d(x, y, [0.01, 0.27, 0.1, 0.072, -0.014, 0.])
    f = d1source + d1sink + d2source + d2sink + d3source + d3sink + d4source + d4sink + d5source + d5sink 
    return f


if __name__ == '__main__':
    lowpass = 0.5
    highpass = 300
    Fs = 30000
    resamp = 12
    tp= 760

    forfilt=np.load('npx_data.npy')

    [b,a] = butter(3, [lowpass/(Fs/2.0), highpass/(Fs/2.0)] ,btype = 'bandpass')
    filtData = filtfilt(b,a, forfilt)
    pots_resamp = filtData[:,::resamp]
    pots = pots_resamp[:, :]
    Fs=int(Fs/resamp)

    pots_for_csd = np.delete(pots, 191, axis=0)
    ele_pos_def = eles_to_coords(np.arange(384,0,-1))
    ele_pos_for_csd = np.delete(ele_pos_def, 191, axis=0)

    k, est_csd, est_pots, ele_pos = npx.do_kcsd(ele_pos_for_csd, pots_for_csd, ele_limit = (0,320))

    csd_at = np.mgrid[0.:0.021:21j,
                      0.:1.:1000j]
    true_csd = gauss_2d_small_f(csd_at)

    cut = 15
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(142)
    set_axis(ax1, -0.05, 1.05, letter= 'B')
    make_plot(ax1, k.estm_x, k.estm_y, est_csd[:,:,tp], ele_pos,
              title='Estimated CSD', cmap='bwr')
    # for i in range(383): plt.text(ele_pos_for_csd[i,0], ele_pos_for_csd[i,1]+8, str(i+1))
    plt.axvline(k.estm_x[cut][0], ls='--', color ='grey', lw=2)

    ax2 = plt.subplot(141)
    set_axis(ax2, -0.05, 1.05, letter= 'A')
    make_plot(ax2, k.estm_x, k.estm_y, est_pots[:,:,tp], ele_pos,
              title='Estimated LFP', cmap='PRGn', ylabel=True)

    ax3 = plt.subplot(143)
    set_axis(ax3, -0.05, 1.05, letter= 'C')
    make_plot(ax3, k.estm_x, k.estm_y, true_csd, ele_pos,
                  title='Model CSD', cmap='bwr')

    eigensources = calculate_eigensources(k)
    projection = csd_into_eigensource_projection(true_csd.flatten(), eigensources)

    ax4 = plt.subplot(144)
    set_axis(ax4, -0.05, 1.05, letter= 'D')
    make_plot(ax4, k.estm_x, k.estm_y, projection.reshape(true_csd.shape), ele_pos,
                  title='Projection', cmap='bwr')
    plt.tight_layout()
    plt.savefig('Neuropixels_with_fitted_model.png', dpi=300)