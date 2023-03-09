from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import copy

def env_data(data, 
               fs=2048, 
               order=4, 
               l_bpf=40, 
               h_bpf=900, 
               lpf_cut=.2):
    """
    Data is band-pass filtered, rectified, and low-pass filtered,
    resulting in a rectified envelope of the signal.
    
    Args:
    	data	        : numpy.ndarray
            1D array containing EMG data to be processed
        fs		        : float
            sampling frequency (Hz)
        order	        : int
            order of filter
        l_bpf	        : float
            lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	        : float
            higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	        : float
            cutoff frequency of the low-pass filter (Hz)
    
    Returns:
        envelope_data   : numpy.ndarray
            1D array containing rectified envelope of the EMG data
    
    Example:
        env_gl_10 = env_data(gl_10_flatten[0])
    """
    
    # Bandpass filter
    b0, a0 = butter(order, [l_bpf, h_bpf], fs=fs, btype="band")
    bpfiltered_data = lfilter(b0, a0, data)
    # Rectifying signal
    rectified_data = abs(bpfiltered_data)
    # Lowpass filter
    b1, a1 = butter(order, lpf_cut, fs=fs, btype="low")
    envelope_data = lfilter(b1, a1, rectified_data)
    
    return envelope_data



def plot_env_data(data, 
                    fs=2048, 
                    order=4, 
                    l_bpf=40, 
                    h_bpf=900, 
                    lpf_cut=.2):
    """
    Plots envelope of data.
    
    Args:
    	data	: numpy.ndarray
            1D array containing EMG data to be processed
        fs		: float
            sampling frequency (Hz)
        order	: int
            order of filter
        l_bpf	: float
            lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	: float
            higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	: float
            cutoff frequency of the low-pass filter (Hz)
    
    Example:
        plot_lpf_signal(gl_10_flatten[0])
    """
    
    envelope_data = env_data(data, fs, order, l_bpf, h_bpf, lpf_cut)
    
    # Plotting results
    x = np.arange(0, len(envelope_data))
    time = x / fs
    plt.rcParams['figure.figsize'] = [15,5]
    plt.plot(time, envelope_data)
    plt.show()
    


def acquire_remove_ind(data,
                       fs=2048,
                       order=4, 
                       l_bpf=40, 
                       h_bpf=900, 
                       lpf_cut=.2, 
                       tol=5e-6):
    """
    Retrieves indices of ramps in-between flat EMG signal that are to be removed.
    data is band-pass filtered, rectified, and low-pass filtered; 
    then starting and end points of ramps are marked.
    
    Args:
    	data	: numpy.ndarray
            1D array containing EMG data to be processed
        fs		: float
            sampling frequency (Hz)
        order	: int
            order of filter
        l_bpf	: float
            lower cutoff frequency of the band-pass filter (Hz)
        h_bpf	: float
            higher cutoff frequency of the band-pass filter (Hz)
        lpf_cut	: float
            cutoff frequency of the low-pass filter (Hz)
        tol		: float
            tolerated value of gradient ~ 0 for marking the start/end of a ramp
    
    Returns:
    	remove_ind	: numpy.ndarray
            contains the indices of the parts to be removed:
            - between data[remove_ind[0]] and data[remove_ind[1]]
            - between data[remove_ind[2]] and data[remove_ind[3]]
    
    Example:
        acquire_remove_ind(gl_10_flatten[0])
    """
    
    envelope_data = env_data(data, fs, order, l_bpf, h_bpf, lpf_cut)
    
    # Plotting results
    x = np.arange(0, len(envelope_data))
    time = x / fs
    plt.rcParams['figure.figsize'] = [15,5]
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.plot(time, envelope_data)
    
    # Finding plateau and start/end of ramp (grad ~ 0)
    envelope_grad = np.gradient(envelope_data)
    flat_ind = np.argwhere(abs(envelope_grad)<=tol)
    
    thr = .25 * (np.max(envelope_data) - np.min(envelope_data))
    
    # Finding indices of start and end of ramps
    ramp_ind = np.array([], dtype="int64")
    for i in range(flat_ind.shape[0] - 1):
        if abs(envelope_data[flat_ind[i+1][0]] - envelope_data[flat_ind[i][0]]) > thr:
            ramp_ind = np.append( ramp_ind, 
                                 [flat_ind[i], flat_ind[i+1]] )
    # Remove duplicate points
    tmp = np.asarray([], dtype="int64")
    for i in range(ramp_ind.shape[0] - 1):
        if ( ramp_ind[i+1] - ramp_ind[i] < 5e2 or 
            (envelope_data[ramp_ind[i]] < thr and envelope_data[ramp_ind[i+1]] < thr ) ): 
            tmp = np.append(tmp, i)
    ramp_ind = np.delete(ramp_ind, tmp)
    
    # Marking start and end of ramp
    plt.scatter(time[flat_ind], envelope_data[flat_ind], c='r', s=40)
    plt.scatter(time[ramp_ind], envelope_data[ramp_ind], c='g', s=100)
    plt.show()
    
    # Indices to remove from the signal
    rm_ind = np.asarray([], dtype="int64")
    for i in range(1, len(ramp_ind) - 1):
        if ( (envelope_data[ramp_ind[i+1]] - envelope_data[ramp_ind[i]] > thr) and 
            (envelope_data[ramp_ind[i+1]] - envelope_data[ramp_ind[i]] > thr) ):
            rm_ind = np.append(rm_ind, [ramp_ind[i - 1], ramp_ind[i + 1]])
    
    # Marking parts to remove
    plt.plot(time, envelope_data)
    plt.scatter(time[rm_ind], envelope_data[rm_ind], c='r', s=40)
    plt.show()

    return rm_ind
    


def modify_signal(data, remove_ind):
    """
    Removes the following elements of data:
        data[remove_ind[0] : remove_ind[1]]
        data[remove_ind[2] : remove_ind[3]]
        ...
    and returns the remaining elements as data_mod.
    
    Args:
    	data	    : numpy.ndarray
            (13, 5) array containing 64 channels of EMG data to be modified
            
    Returns:
        data_mod    : numpy.ndarray
            (13, 5) array containing modified EMG data
        
    Example:
        modify_signal(gl_10, remove_ind_gl10)
    """
    
    data_mod = copy.deepcopy(data)
    
    # Acquiring indices to remove from data
    rm_ind_flt = np.ndarray.flatten(remove_ind)
    indices_to_remove = np.asarray([], dtype="int64")
    for i in range(len(rm_ind_flt) // 2):
        indices_to_remove = np.append( indices_to_remove, 
                                        [ np.arange(rm_ind_flt[2*i], rm_ind_flt[2*i + 1]) ] )
    
    # Removing indices from data
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if len(data[i][j]) != 0:
                tmp = np.delete(data[i][j][0], indices_to_remove)
                data_mod[i][j] = np.asarray([tmp]) 
    return data_mod


def crop_data(data, start=0.0, end=40.0, fs=2048):
    """
    Returns the cropped data between start and end (in seconds).
    
    Args:
    	data	    : numpy.ndarray
            (13, 5) array containing 64 channels of EMG data to be modified
        start       : float
            starting time (in seconds) of the cropped data
        end         : float
            end time (in seconds) of the cropped data
        fs          : int
            sampling frequency (Hz)
            
    Returns:
        data_mod    : numpy.ndarray
            (13, 5) array containing modified EMG data
        
    Example:
        gl_10_crop = crop_data(gl_10, end=70)
    """
    if end > (len(data[0][1][0]) / fs):
        end = len(data[0][1][0]) / fs
    data_crop = copy.deepcopy(data)
    for i in range(0, data_crop.shape[0]):
        for j in range(0, data_crop.shape[1]):
            if len(data_crop[i][j]) != 0:
                data_crop[i][j] = np.asarray([data[i][j][0][int(start*fs): int(end*fs)]])
    return data_crop
    


    
def visualize_pt(ind_pt, data, fs=2048, title="decomposition"):
    """
    Plots envelope of data and pulse trains of motor units from decomp.
    
    Args:
    	ind_pt  : numpy.ndarray
            indices of motor units' pulse trains
        data	: numpy.ndarray
            1D array containing EMG data to be processed
        fs		: float
            sampling frequency (Hz)
    
    Example:
        visualize_pt(decomp_gl_10_mod, gl_10_mod)
    """

    n_mu = ind_pt.shape[0]
    x = data[0][1].shape[1]
    
    # Pulse train
    pt = np.zeros((n_mu, x), dtype="int64")
    for i in range(ind_pt.shape[0]):
        for j in range(ind_pt[i].shape[0]):
            tmp = ind_pt[i][j]
            pt[i][tmp] = 1
    
    # Creating subplot
    n_rows = ind_pt.shape[0] + 1
    height_ratio = np.ones(n_rows)
    height_ratio[0] = 5
    plt.rcParams['figure.figsize'] = [35, 10+(2.5*(n_rows-1))]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    # Plotting envelope of emg_data
    envelope_data = env_data(data[0][1][0])
    x = np.arange(0, len(envelope_data), dtype="float")
    time = x / float(fs)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    # plt.rcParams['figure.figsize'] = [35,10]
    ax[0].plot(time, envelope_data)
    ax[0].set_title(title, fontsize=24)

    for i in range(1, n_rows):
        y = pt[i-1]
        ax[i].plot(time,y)
        ax[i].set_ylabel(f"MU {i-1}", fontsize=20)
    plt.show()


def visualize_pt_window(ind_pt, data, start=0.0, end=40.0, fs=2048, title="decomposition"):
    """
    Plots a window of the envelope of data and pulse trains of motor units from decomp.
    
    Args:
    	ind_pt  : numpy.ndarray
            indices of motor units' pulse trains
        data	: numpy.ndarray
            1D array containing EMG data to be processed
        start       : float
            starting time (in seconds) of the window
        end         : float
            end time (in seconds) of the window
        fs		: float
            sampling frequency (Hz)
    
    Example:
        visualize_pt(decomp_gl_10_mod, gl_10_mod)
    """

    # Windowed data
    data_crop = crop_data(data, start = start, end = end)
    n_mu = ind_pt.shape[0]
    x = data_crop[0][1].shape[1]
    
    # Pulse train in the range of the window
    pt = np.zeros((n_mu, x), dtype="int64")
    for i in range(ind_pt.shape[0]):
        for j in range(ind_pt[i].shape[0]):
            if ind_pt[i][j] < x:
                tmp = ind_pt[i][j]
                pt[i][tmp] = 1
    
    # Creating subplot
    n_rows = ind_pt.shape[0] + 1
    height_ratio = np.ones(n_rows)
    height_ratio[0] = 5
    plt.rcParams['figure.figsize'] = [35, 10+(2.5*(n_rows-1))]
    fig, ax = plt.subplots(n_rows , 1, gridspec_kw={'height_ratios': height_ratio})
    
    # Plotting envelope of cropped emg data
    envelope_data = env_data(data_crop[0][1][0])
    x = np.arange(0, len(envelope_data), dtype="float")
    time = x / float(fs)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    # plt.rcParams['figure.figsize'] = [35,10]
    ax[0].plot(time, envelope_data)
    ax[0].set_title(title, fontsize=24)
    
    # Plotting pulse trains
    for i in range(1, n_rows):
        y = pt[i-1]
        ax[i].plot(time,y)
        ax[i].set_ylabel(f"MU {i-1}", fontsize=20)
    plt.show()
   
    
def cross_corr(muap_dict_1, muap_dict_2):
    """
    Calculates the cross correlation between MUAPs of 2 decompositions, 
    muap_dict_1 and muap_dict_2
    
    Args:
        muap_dict_1 : dictionary of MUAP shapes for each motor unit
        muap_dict_2 : dictionary of MUAP shapes for each motor unit
    
    Returns:
        cc_values   : numpy array of cross correlation values
    
    Example:
        cc_gl10_gl30 = cross_corr(muap_gl10, muap_gl30)
    """
    
    # number of channels
    n_ch = np.max(muap_dict_1["mu_0"]["channel"]) + 1
    
    # number of motor units 
    n_mu_1 = len(muap_dict_1)
    n_mu_2 = len(muap_dict_2)

    # length of MUAP shape
    muap_size = int(muap_dict_1["mu_0"]["signal"].shape[0] / n_ch)
    
    # Initializing array to store cross correlation values
    cc_values = np.zeros((n_mu_1, n_mu_2, n_ch), dtype="float")
    
    # Comparing each MUAP (k=0-63) for every Motor Unit in muap_1 (i) against 
    #           each MUAP (k=0-63) for every Motor Unit in muap_2 (j) 
    for i in range(n_mu_1):
        for j in range(n_mu_2):
            for k in range(n_ch):
                # Normalized MUAP signal from muap_dict_1
                muap_1 = muap_dict_1[f"mu_{i}"]["signal"][muap_size*k : muap_size*(k+1)]
                muap_1_norm = (muap_1 - np.mean(muap_1)) / (np.std(muap_1) * len(muap_1))
                
                # Normalized MUAP signal from muap_dict_2
                muap_2 = muap_dict_2[f"mu_{j}"]["signal"][muap_size*k : muap_size*(k+1)]
                muap_2_norm = (muap_2 - np.mean(muap_2)) / np.std(muap_2)
                
                # Cross correlation
                cc_muap = np.correlate(muap_1_norm, muap_2_norm)
                
                # Store value in array
                cc_values[i][j][k] = cc_muap
                
    return cc_values
    
    
    
def mean_cc(cc_values):
    """
    Calculates mean value of cross correlation between MUAPs across channels (64)
    from a known array of cross correlation values (cc_values)
        
    Args:
        cc_values       : numpy.ndarray
            (n_mu_1, n_mu_2, n_ch) array containing cross correlation values between
            MUAPs from MU1 and MU2
            
    Returns:
        mean_cc_values   : numpy.ndarray
            (n_mu_1, n_mu_2) array containing mean values of cross correlation between
            MUAPs from MU1 and MU2

    Example:
        avg_cc_gl10_gl30 = avg_cross_corr(cc_gl10_gl30)     
    """
    
    mean_cc_values = np.mean(cc_values, axis=2, dtype="float64")
    
    return mean_cc_values
    


def find_high_cc(mean_cc_values, decomp1="decomposition_1", decomp2="decomposition_2", thr=.75):
    """
    Retrieves indices of where mean_cc_values > thr;
    and prints these values and indices
    
    Args:
        mean_cc_values  : numpy.ndarray
            (n_mu_1, n_mu_2) array containing mean values of cross correlation between
            MUAPs from MU1 and MU2
        thr             : float
            threshold value
    
    Returns:
        high_cc_values  : numpy.ndarray
            array containing indices where mean_cc_values > thr
    """
    high_cc_values = np.argwhere(mean_cc_values > thr)

    index_max = np.unravel_index(np.argmax(mean_cc_values), mean_cc_values.shape)
    
    print("mean cc_values for", decomp1,"and", decomp2, "higher than", thr, ":",)
    print(np.max(mean_cc_values), "at", index_max)
    for i in range(high_cc_values.shape[0]):
        tmp1 = high_cc_values[i][0]
        tmp2 = high_cc_values[i][1]
        print(mean_cc_values[tmp1][tmp2], "at", high_cc_values[i])
    return high_cc_values



def plot_meancc(mean_cc_values, y_axis="decomposition_1", x_axis="decomposition_2"):
    """
    Plots mean cross correlation between motor units from 2 decompositions.
    
    Args:
    	mean_cc_values	: numpy.ndarray
            2D array containing mean cross correlation values
        y_axis  		: char
            name of y axis
        x_axis      	: char
            name of x axis
            
    Example:
        plot_meancc(mean_cc_gl10_gl10_mod)
    """
    
    # make plot
    fig, ax = plt.subplots()
    ratio = mean_cc_values.shape[1] / mean_cc_values.shape[0]
    plt.rcParams["figure.figsize"] = [10*ratio, 10]

    # show image
    shw = ax.imshow(mean_cc_values)

    # make bar
    bar = plt.colorbar(shw, fraction=0.046, pad=0.04)

    # show plot with labels
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel(y_axis, fontsize=12)
    bar.set_label('Mean cross-correlation', fontsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.show()