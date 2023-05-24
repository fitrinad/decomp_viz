# Import emgdecompy functions
from emgdecompy.decomposition import *
from emgdecompy.contrast import *
from emgdecompy.viz import *
from emgdecompy.preprocessing import *

from scipy import linalg
from scipy.signal import find_peaks
from scipy.stats import variation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from functions import *


#############################################################################################################################
### Training module  ########################################################################################################
#############################################################################################################################
"""
Decomposition functions for the training module from:
https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
"""
def silhouette_score_tmod(s_i, peak_indices):
    """
    Silhouette score function from: 
    https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
    Calculates silhouette score on the estimated source.

    Defined as the difference between within-cluster sums of point-to-centroid distances
    and between-cluster sums of point-to-centroid distances.
    Measure is normalized by dividing by the maximum of these two values (Negro et al. 2016).

    Parameters
    ----------
        s_i: numpy.ndarray
            Estimated source. 1D array containing K elements, where K is the number of samples.
        peak_indices_a: numpy.ndarray
            1D array containing the peak indices.

    Returns
    -------
        float
            Silhouette score.

    Examples
    --------
    >>> s_i = np.array([0.80749775, 10, 0.49259282, 0.88726069, 5,
                        0.86282998, 3, 0.79388539, 0.29092294, 2])
    >>> peak_indices = np.array([1, 4, 6, 9])
    >>> silhouette_score(s_i, peak_indices)
    0.740430148513959

    """
    # Create clusters
    peak_cluster = s_i[peak_indices]
    noise_cluster = np.delete(s_i, peak_indices)

    # Create centroids
    peak_centroid = peak_cluster.mean()
    noise_centroid = noise_cluster.mean()

    # Calculate within-cluster sums of point-to-centroid distances
    intra_sums = (
        abs(peak_cluster - peak_centroid).sum()
        + abs(noise_cluster - noise_centroid).sum()
    )

    # Calculate between-cluster sums of point-to-centroid distances
    inter_sums = (
        abs(peak_cluster - noise_centroid).sum()
        + abs(noise_cluster - peak_centroid).sum()
    )

    diff = inter_sums - intra_sums

    sil = diff / max(intra_sums, inter_sums)

    return sil, peak_centroid, noise_centroid



def refinement_tmod(
    w_i, z, i, l=31, sil_pnr=True, thresh=0.9, max_iter=10, random_seed=None, verbose=False
):
    """
    Refinement function from: 
    https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
    Refines the estimated separation vectors determined by the `separation` function
    as described in Negro et al. (2016). Uses a peak-finding algorithm combined
    with K-Means clustering to determine the motor unit spike train. Updates the 
    estimated separation vector accordingly until regularity of the spike train is
    maximized. Steps 4, 5, and 6 in Negro et al. (2016).

    Parameters
    ----------
        w_i: numpy.ndarray
            Current separation vector to refine.
        z: numpy.ndarray
            Centred, extended, and whitened EMG data.
        i: int
            Decomposition iteration number.
        l: int
            Required minimal horizontal distance between peaks in peak-finding algorithm.
            Default value of 31 samples is approximately equivalent
            to 15 ms at a 2048 Hz sampling rate.
        sil_pnr: bool
            Whether to use SIL or PNR as acceptance criterion.
            Default value of True uses SIL.
        thresh: float
            SIL/PNR threshold for accepting a separation vector.
        max_iter: int > 0
            Maximum iterations for refinement.
        random_seed: int
            Used to initialize the pseudo-random processes in the function.
        verbose: bool
           If true, refinement information is printed.

    Returns
    -------
        numpy.ndarray
            Separation vector if SIL/PNR is above threshold.
            Otherwise return empty vector.
        numpy.ndarray
            Estimated source obtained from dot product of separation vector and z.
            Empty array if separation vector not accepted.
        numpy.ndarray
            Peak indices for peaks in cluster "a" of the squared estimated source.
            Empty array if separation vector not accepted.
        float
            Silhouette score if SIL/PNR is above threshold.
            Otherwise return 0.
        float
            Pulse-to-noise ratio if SIL/PNR is above threshold.
            Otherwise return 0.

    Examples
    --------
    >>> w_i = refinement(w_i, z, i)
    """
    cv_curr = np.inf # Set it to inf so there isn't a chance the loop breaks too early

    for iter in range(max_iter):
        
        w_i = normalize(w_i) # Normalize separation vector

        # a. Estimate the i-th source
        s_i = np.dot(w_i, z)  # w_i and w_i.T are equal

        # Estimate pulse train pt_n with peak detection applied to the square of the source vector
        s_i2 = np.square(s_i)

        # Peak-finding algorithm
        peak_indices, _ = find_peaks(
            s_i2, distance=l
        )

        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2, random_state=random_seed)
        kmeans.fit(s_i2[peak_indices].reshape(-1, 1))
        
        # Determine which cluster contains large peaks
        centroid_a = np.argmax(
            kmeans.cluster_centers_
        )
        
        # Determine which peaks are large (part of cluster a)
        peak_a = ~kmeans.labels_.astype(
            bool
        )

        if centroid_a == 1: # If cluster a corresponds to kmeans label 1, change indices correspondingly
            peak_a = ~peak_a

        
        # Get the indices of the peaks in cluster a
        peak_indices_a = peak_indices[
            peak_a
        ]

        # c. Update inter-spike interval coefficients of variation
        isi = np.diff(peak_indices_a)  # inter-spike intervals
        cv_prev = cv_curr
        cv_curr = variation(isi)

        if np.isnan(cv_curr): # Translate nan to 0
            cv_curr = 0

        if (
            cv_curr > cv_prev
        ):
            break
            
        elif iter != max_iter - 1: # If we are not on the last iteration
            # d. Update separation vector for next iteration unless refinement doesn't converge
            j = len(peak_indices_a)
            w_i = (1 / j) * z[:, peak_indices_a].sum(axis=1)

    # If silhouette score is greater than threshold, accept estimated source and add w_i to B
    sil, peak_centroid, noise_centroid = silhouette_score_tmod(
        s_i2, peak_indices_a
    )
    pnr_score = pnr(s_i2, peak_indices_a)
    
    if isi.size > 0 and verbose:
        print(f"Cov(ISI): {cv_curr / isi.mean() * 100}")

    if verbose:
        print(f"PNR: {pnr_score}")
        print(f"SIL: {sil}")
        print(f"cv_curr = {cv_curr}")
        print(f"cv_prev = {cv_prev}")
        
        if cv_curr > cv_prev:
            print(f"Refinement converged after {iter} iterations.")

    if sil_pnr:
        score = sil # If using SIL as acceptance criterion
    else:
        score = pnr_score # If using PNR as acceptance criterion
    
    # Don't accept if score is below threshold or refinement doesn't converge
    if score < thresh or cv_curr < cv_prev or cv_curr == 0: 
        w_i = np.zeros_like(w_i) # If below threshold, reject estimated source and return nothing
        return w_i, np.zeros_like(s_i), np.array([]), 0, 0, 0, 0
    else:
        print(f"Extracted source at iteration {i}.")
        return w_i, s_i, peak_indices_a, sil, pnr_score, peak_centroid, noise_centroid


def decomposition_tmod(
    x,
    discard=None,
    R=16,
    M=64,
    bandpass=True,
    lowcut=10,
    highcut = 900,
    fs=2048,
    order=6,
    Tolx=10e-4,
    contrast_fun=skew,
    ortho_fun=gram_schmidt,
    max_iter_sep=10,
    l=31,
    sil_pnr=True,
    thresh=0.9,
    max_iter_ref=10,
    random_seed=None,
    verbose=False
):
    """
    Decomposition function from: 
    https://github.com/The-Motor-Unit/EMGdecomPy/tree/main/src/emgdecompy
    Blind source separation algorithm that utilizes the functions
    in EMGdecomPy to decompose raw EMG data. Runs data pre-processing, separation,
    and refinement steps to extract individual motor unit activity from EMG data. 
    Runs steps 1 through 6 in Negro et al. (2016).

    Parameters
    ----------
        x: numpy.ndarray
            Raw EMG signal.
        discard: slice, int, or array of ints
            Indices of channels to discard.
        R: int
            How far to extend x.
        M: int
            Number of iterations to run decomposition for.
        bandpass: bool
            Whether to band-pass filter the raw EMG signal or not.
        lowcut: float
            Lower range of band-pass filter.
        highcut: float
            Upper range of band-pass filter.
        fs: float
            Sampling frequency in Hz.
        order: int
            Order of band-pass filter. 
        Tolx: float
            Tolerance for element-wise comparison in separation.
        contrast_fun: function
            Contrast function to use.
            skew, og_cosh or exp_sq
        ortho_fun: function
            Orthogonalization function to use.
            gram_schmidt or deflate
        max_iter_sep: int > 0
            Maximum iterations for fixed point algorithm.
        l: int
            Required minimal horizontal distance between peaks in peak-finding algorithm.
            Default value of 31 samples is approximately equivalent
            to 15 ms at a 2048 Hz sampling rate.
        sil_pnr: bool
            Whether to use SIL or PNR as acceptance criterion.
            Default value of True uses SIL.
        thresh: float
            SIL/PNR threshold for accepting a separation vector.
        max_iter_ref: int > 0
            Maximum iterations for refinement.
        random_seed: int
            Used to initialize the pseudo-random processes in the function.
        verbose: bool
            If true, decomposition information is printed.

    Returns
    -------
        dict
            Dictionary containing:
                B: numpy.ndarray
                    Matrix whose columns contain the accepted separation vectors.
                MUPulses: numpy.ndarray
                    Firing indices for each motor unit.
                SIL: numpy.ndarray
                    Corresponding silhouette scores for each accepted source.
                PNR: numpy.ndarray
                    Corresponding pulse-to-noise ratio for each accepted source.

    Examples
    --------
    >>> gl_10 = loadmat('../data/raw/gl_10.mat')
    >>> x = gl_10['SIG']
    >>> decomposition(x)
    """

    # Flatten
    x = flatten_signal(x)
    
    # Discard unwanted channels
    if discard is not None:
        x = np.delete(x, discard, axis=0)

    # Apply band-pass filter
    if bandpass:
        x = np.apply_along_axis(
            butter_bandpass_filter,
            axis=1,
            arr=x,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs, 
            order=order)

    # Center
    x = center_matrix(x)

    print("Centred.")

    # Extend
    x_ext = extend_all_channels(x, R)

    print("Extended.")

    # Whiten
    z = whiten(x_ext)

    print("Whitened.")

    decomp_results = {}  # Create output dictionary

    B = np.zeros((z.shape[0], z.shape[0]))  # Initialize separation matrix
    
    z_peak_indices, z_peak_heights = initial_w_matrix(z)  # Find highest activity columns in z
    z_peaks = z[:, z_peak_indices] # Index the highest activity columns in z

    MUPulses = []
    sils = []
    pnrs = []
    peak_centroids = []
    noise_centroids = []
    s = []


    for i in range(M):

        z_highest_peak = (
            z_peak_heights.argmax()
        )  # Determine which column of z has the highest activity

        w_init = z_peaks[
            :, z_highest_peak
        ]  # Initialize the separation vector with this column

        if verbose and (i + 1) % 10 == 0:
            print(i)

        # Separate
        w_i = separation(
            z, w_init, B, Tolx, contrast_fun, ortho_fun, max_iter_sep, verbose
        )

        # Refine
        w_i, s_i, mu_peak_indices, sil, pnr_score, peak_centroid, noise_centroid = refinement_tmod(
            w_i, z, i, l, sil_pnr, thresh, max_iter_ref, random_seed, verbose
        )
    
        B[:, i] = w_i # Update i-th column of separation matrix

        if mu_peak_indices.size > 0:  # Only save information for accepted vectors
            MUPulses.append(mu_peak_indices)
            sils.append(sil)
            pnrs.append(pnr_score)
            peak_centroids.append(peak_centroid)
            noise_centroids.append(noise_centroid)
            s.append(s_i)

        # Update initialization matrix for next iteration
        z_peaks = np.delete(z_peaks, z_highest_peak, axis=1)
        z_peak_heights = np.delete(z_peak_heights, z_highest_peak)
        
    decomp_results["B"] = B[:, B.any(0)] # Only save columns of B that have accepted vectors
    decomp_results["MUPulses"] = np.array(MUPulses, dtype="object")
    decomp_results["SIL"] = np.array(sils)
    decomp_results["PNR"] = np.array(pnrs)
    decomp_results["peak_centroids"] = np.array(peak_centroids)
    decomp_results["noise_centroids"] = np.array(noise_centroids)
    decomp_results["s"] = np.array(s)

    return decomp_results


def calc_centroids(B, x, random_seed=None, discard=None, R=16, l=31):
    if (x[0][0].size == 0 and x.ndim == 2) or x.ndim == 3:
        x = flatten_signal(x)
    # Discard unwanted channels
    if discard is not None:
        x = np.delete(x, discard, axis=0)

    x = center_matrix(x)
    x_ext = extend_all_channels(x, R=R)
    z = whiten(x_ext)

    s = np.dot(B.T, z)

    peak_a_centroids = []
    noise_centroids = []

    for i in range(s.shape[0]):
        s2_i = np.square(s[i])

        # Clustering
        # Peak-finding algorithm
        peak_indices, _ = find_peaks(s2_i, distance=l)

        # b. Use KMeans to separate large peaks from relatively small peaks, which are discarded
        kmeans = KMeans(n_clusters=2, random_state=random_seed)
        kmeans.fit(s2_i[peak_indices].reshape(-1, 1))
        
        # Determine which cluster contains large peaks
        centroid_a = np.argmax(kmeans.cluster_centers_)
        # Determine which peaks are large (part of cluster a)
        peak_a = kmeans.labels_.astype(bool)
        if centroid_a == 0:
            peak_a = ~peak_a
        
        # Get the indices of the peaks in cluster a
        peak_a_indices = peak_indices[peak_a]

        # peak_a cluster and noise cluster
        peak_a_cluster = s2_i[peak_a_indices]
        noise_cluster = np.delete(s2_i, peak_a_indices)

        # Centroids
        peak_a_centroid = peak_a_cluster.mean()
        noise_centroid = noise_cluster.mean()

        peak_a_centroids.append(peak_a_centroid)
        noise_centroids.append(noise_centroid)

    peak_a_centroids = np.array(peak_a_centroids)
    noise_centroids = np.array(noise_centroids)

    return peak_a_centroids, noise_centroids


def sep_realtime(x, B, discard=None, center=True, 
                 bandpass=True, lowcut=10, highcut=900, fs=2048, order=6, 
                 R=16):
    """
    Returns matrix containing separation vectors for realtime decomposition.

    Args:
        x           : numpy.ndarray
            Raw EMG signal
        B           : numpy.ndarray
            Matrix containing separation vectors from training module
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x

    Returns:
        B_realtime  : numpy.ndarray
            Separation matrix for realtime decomposition
    """
    # Flatten signal
    if ((x[0][0].size == 0 or 
         x[12][0].size == 0) and x.ndim == 2) or x.ndim == 3:
        x = flatten_signal(x)
    
    # Discarding channels
    if discard is not None:
        x = np.delete(x, discard, axis=0)

    # band-pass filter
    if bandpass:
        x = np.apply_along_axis(
            butter_bandpass_filter,
            axis=1,
            arr=x,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs, 
            order=order)

    # Center
    if center: 
        x_cent = center_matrix(x)
    else:
        x_cent = x

    # Extend
    x_ext = extend_all_channels(x_cent, R)

    # Whitening Matrix: wzca
    #   Calculate covariance matrix
    cov_mat = np.cov(x_ext, rowvar=True, bias=True)
    #   Eigenvalues and eigenvectors
    w, v = linalg.eig(cov_mat)
    #   Apply regularization factor, replacing eigenvalues smaller than it with the factor
    reg_factor = w[round(len(w) / 2):].mean()
    w = np.where(w < reg_factor, reg_factor, w)
    #   Diagonal matrix inverse square root of eigenvalues
    diagw = np.diag(1 / (w ** 0.5))
    diagw = diagw.real
    #   Whitening using zero component analysis: v diagw v.T x
    wzca = np.dot(v, np.dot(diagw, v.T))

    # 1. Realtime separation matrix: 
    #    B_realtime = wzca . B
    B_realtime = np.dot(wzca, B)
    #   Normalized separation matrix
    for i in range(B_realtime.shape[0]):
        B_realtime[i] = normalize(B_realtime[i])

    # 2. Mean of training data
    x_ext_tm = extend_all_channels(x, R=R)
    mean_tm = x_ext_tm.mean(axis=1)

    return B_realtime, mean_tm



#############################################################################################################################
### Decomposition module  ###################################################################################################
#############################################################################################################################
def source_extraction(x, B_realtime, mean_tm=None, discard=None, 
                      bandpass=True, lowcut=10, highcut=900, fs=2048, order=6, 
                      R=16):
    """
    Returns matrix containing source vectors estimation from the EMG signal (x).

    Args:
        x           : numpy.ndarray
            Raw EMG signal
        B_realtime  : numpy.ndarray
            Matrix containing separation vectors for realtime source extraction
        discard     : int or array of ints
            Indices of channels to be discarded
        R           : int
            How far to extend x

    Returns:
        s           : numpy.ndarray
            Matrix containing source vectors
        x_ext       : numpy.ndarray
            Extended EMG signal
    """    

    # Flatten signal
    x = flatten_signal(x)

    # Discarding channels
    if discard is not None:
        x = np.delete(x, discard, axis=0)

    # band-pass filter
    if bandpass:
        x = np.apply_along_axis(
            butter_bandpass_filter,
            axis=1,
            arr=x,
            lowcut=lowcut,
            highcut=highcut,
            fs=fs, 
            order=order)

    # Extend
    x_ext = extend_all_channels(x, R)

    if mean_tm is not None: # Use mean from training module
        # x_ext - mean_tm
        x_ext = x_ext.T - mean_tm.T
        x_ext = x_ext.T
    else: # Use mean from realtime data
        # x_ext - x.mean
        x_ext = center_matrix(x_ext)

    # Source extraction
    s = np.dot(B_realtime.T, x_ext)

    return s, x_ext


def peak_extraction(s, l=31):
    """
    Detects and extracts peaks from the each squared source vector s[i] from matrix s.

    Args:
        x_ext       : numpy.ndarray
            Extended EMG signal
        s           : numpy.ndarray
            Matrix containing source vectors
        l           : int
            Minimal horizontal distance between peaks in the peak-finding algorithm
            (default: l=31, approximately 15 ms for fs = 2048 Hz)
        
    Returns:
        peak_indices  : numpy.ndarray
            Matrix containing separation vectors for realtime decomposition
    """
    # Squared source vectors
    s2 = np.square(s)
    
    peak_indices = []
    # Detecting peaks in s2
    for i in range(s2.shape[0]):
        peak_indices_i , _ = find_peaks(s2[i], distance=l)
        peak_indices.append(peak_indices_i.astype("int64"))

    length = len(peak_indices[0])
    if any(len(arr) != length for arr in peak_indices):
        peak_indices = np.array(peak_indices, dtype="object")
    else:
        peak_indices = np.array(peak_indices, dtype="int64")

    return s2, peak_indices


def dist_ratio(s2_i, signal_cluster, noise_cluster):

    # Centroids
    signal_centroid = signal_cluster.mean()
    noise_centroid = noise_cluster.mean()

    # Calculating distance of each signal peak to signal centroid and noise centroid
    dist_signal = abs(signal_cluster - signal_centroid)
    dist_noise = abs(signal_cluster - noise_centroid)

    # Calculating ratio of distances
    peak_dist_ratio = dist_signal / (dist_noise + dist_signal)

    """
    # Calculating distance of each signal peak to signal centroid and noise centroid
    dist_signal = abs(signal_cluster - signal_centroid).sum()
    dist_noise = abs(signal_cluster - noise_centroid).sum()
    diff = abs(dist_signal - dist_noise)

    # Calculating ratio of distances
    peak_dist_ratio = diff / max(dist_signal, dist_noise)
    """

    return peak_dist_ratio


#############################################################################################################################
### kmeans_sil_cp  ##########################################################################################################
#############################################################################################################################
def sort_peaks(s2, peak_indices, use_kmeans=True, random_seed=None, thd_noise=0.38):
    signal_clusters = []
    signal_centroids = []
    max_signals = []
    noise_clusters = []
    noise_centroids = []
    n_signal = []
    n_noise = []
    peak_indices_signal = []
    peak_indices_noise = []

    for i in range(s2.shape[0]):
        if use_kmeans:
            # Separating large peaks from relatively small peaks (noise)
            kmeans = KMeans(n_clusters=2, random_state=random_seed)
            kmeans.fit(s2[i][peak_indices[i]].reshape(-1,1))

            # Signal cluster centroid (sc_i)
            sc_i = np.argmax(kmeans.cluster_centers_)
            # Determining which cluster contains large peaks (signal)
            signal_indices = kmeans.labels_.astype(bool) # if sc_i == 1
            if sc_i == 0:
                signal_indices = ~signal_indices      
        else:
            # noise: peaks < thd_noise*s2.max()
            signal_indices = s2[i][peak_indices[i]] > thd_noise*s2[i].max()
            # signal_indices = signal_indices.flatten()
        n_signal_idx = signal_indices.sum()
        n_noise_idx = (~signal_indices).sum()
        
        # Indices of the peaks in signal cluster
        peak_indices_signal_i = peak_indices[i][signal_indices]
        peak_indices_noise_i = peak_indices[i][~signal_indices]
        peak_indices_signal.append(peak_indices_signal_i)
        peak_indices_noise.append(peak_indices_noise_i)

        # Signal cluster and Noise cluster
        signal_cluster = s2[i][peak_indices_signal_i]
        noise_cluster = np.delete(s2[i], peak_indices_signal_i)

        # Centroids
        signal_centroid = signal_cluster.mean()
        noise_centroid = noise_cluster.mean()
        
        signal_clusters.append(signal_cluster)
        signal_centroids.append(signal_centroid)
        max_signals.append(signal_cluster.max())
        noise_clusters.append(noise_cluster)
        noise_centroids.append(noise_centroid)
        n_signal.append(n_signal_idx)
        n_noise.append(n_noise_idx)

    n_signal = np.array(n_signal, dtype="int")
    n_noise = np.array(n_noise, dtype="int")
    peak_indices_signal = np.array(peak_indices_signal, dtype="object")
    peak_indices_noise = np.array(peak_indices_noise, dtype="object")
    
    signal_centroids = np.array(signal_centroids, dtype="float")
    max_sc = signal_centroids.max()
    signal_centroids = signal_centroids / max_sc
    signal_clusters = np.array(signal_clusters, dtype="object")
    signal_clusters = signal_clusters / max_sc
    max_signals = np.array(max_signals, dtype="float")
    max_signals = max_signals / max_sc
    
    noise_clusters = np.array(noise_clusters, dtype="object")
    noise_clusters = noise_clusters / max_sc
    noise_centroids = np.array(noise_centroids, dtype="float")
    noise_centroids = noise_centroids / max_sc
    
    # Distance between centroids
    centroid_dists = signal_centroids - noise_centroids 
    s2 = s2 / max_sc

    signal = {"n_signal": n_signal, 
            "peak_indices_signal": peak_indices_signal, 
            "signal_clusters": signal_clusters,
            "signal_centroids": signal_centroids}
    noise = {"n_noise": n_noise, 
            "peak_indices_noise": peak_indices_noise, 
            "noise_clusters": noise_clusters,
            "noise_centroids": noise_centroids}

    return signal, noise, centroid_dists, s2, max_sc



def spike_classification(s2, peak_indices,
                         use_kmeans=True, random_seed=None, thd_noise=0.38, 
                         classify_mu=True, sil_dist=False, 
                         thd_sil=0.9, thd_cent_dist=0.6,
                         sc_tm=None, nc_tm=None):
    """
    Returns a matrix of motor unit pulses.

    Args:
        s2              : numpy.ndarray
            Matrix containing squared source vectors
        peak_indices    : numpy.ndarray
            Matrix containing indices of detected peaks from s2
        use_kmeans      : bool
            Separates large peaks from small peaks using kmeans clustering if True
        random_seed     : int
            Used to initialize the pseudo-random processes in the function
        thd_sil         : float
            Threshold of the silhouette score
        thd_dist_ratio  : float
            Threshold of the peak distance ratio
        sil_dist        : bool
            Classifies peaks as motor unit pulses according to the silhouette score (if True) 
            or peak distance ratio (if False) 
    
    Returns:
        MUPulses        : numpy.ndarray
            Matrix containing indices of motor unit pulses
    """
    
    signal, noise, centroid_dists, s2, _ = sort_peaks(s2=s2, peak_indices=peak_indices, 
                                                           use_kmeans=use_kmeans, random_seed=random_seed, 
                                                           thd_noise=thd_noise)

    peak_indices_signal = signal["peak_indices_signal"]
    MUPulses = []
    sil_scores = []
    cent_dists = []
    for i in range(s2.shape[0]):
        add_peaks = peak_indices_signal[i]
        if classify_mu: # whether MU i is firing or not, based on SIL or centroid distance
            add_peaks = peak_indices_signal[i]
            if sil_dist:
                # Silhouette score
                sil = silhouette_score(s2[i], peak_indices_signal[i])
                sil_scores.append(sil)
                if sil < thd_sil:
                    add_peaks = []
            else:
                cent_dist = centroid_dists[i] 
                cent_dists.append(cent_dist)
                if cent_dist < thd_cent_dist:
                    add_peaks = []

        # Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
        # adding peaks closer to signal centroid
        if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):
            # s2_max = s2[i][add_peaks].max()
            # add_indices = ( abs(s2[i][add_peaks]/s2_max - nc_tm[i]) > 
            #                abs(s2[i][add_peaks]/s2_max - sc_tm[i]) )
            # add_peaks = add_peaks[add_indices] 
            add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i])) > 
                            abs(s2[i][add_peaks] - (sc_tm[i])) )
            add_peaks = add_peaks[add_indices] 
        
        MUPulses.append(np.array(add_peaks, dtype="int64"))
    
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")
        
    cls_values = {"centroid_dists": np.array(cent_dists, dtype="float"),
                  "sil_scores": np.array(sil_scores, dtype="float")}

    return MUPulses, cls_values, signal, noise, s2


def batch_sil(sil_scores_prev, sil_scores, overlap=1.0, fs=2048):
    n_mu = sil_scores_prev.shape[0]
    end_1 = sil_scores_prev.shape[1]
    end_2 = sil_scores.shape[1]
    ol = int(overlap * fs)
    curr_size = end_1 + end_2 - ol
    if end_2 < ol:
        curr_size = end_1
    sil_scores_curr = np.zeros((n_mu, curr_size), dtype="float")
    
    sil_scores_curr[:, :end_1] = sil_scores_prev
    if end_2 <= ol:
        sil_scores_curr[:, end_1-end_2:curr_size] = sil_scores_prev[:, end_1-end_2:end_1]
    else:
        sil_scores_curr[:, end_1:curr_size] = sil_scores[:, ol:end_2]
        """if ol!=0:
            for i in range(n_mu):
                tmp = np.arange(sil_scores_prev[i][end_1-1-ol],
                                sil_scores[i][ol],
                                (sil_scores[i][ol]-sil_scores_prev[i][end_1-1-ol])/ol)
                sil_scores_curr[i, end_1-ol:end_1] = tmp"""
    return sil_scores_curr


def batch_decomp(data, B_realtime, mean_tm=None, discard=None, l=31,
                 use_kmeans=False, thd_noise=0.38,
                 classify_mu=True, sil_dist = True,
                 thd_sil=0.9, thd_cent_dist=0.6,
                 sc_tm=None, nc_tm=None,
                 batch_size=4.0, overlap=0.0, fs=2048):
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        end_time = data[0][1].shape[1] / fs
    else:
        end_time = data.shape[1] / fs

    time = 0.0
    while True:
        raw = crop_data(data, start = time, end = time+batch_size)

        # Source extraction
        s, _ = source_extraction(x = raw, 
                                 B_realtime = B_realtime, 
                                 mean_tm = mean_tm,
                                 discard=discard)

        # Peak extraction
        s2, peak_indices = peak_extraction(s=s, l=l)
        
        # Spike classification
        if time == 0.0:
            MUPulses, cls_values, _, _, _ = spike_classification(s2=s2, 
                                                        peak_indices=peak_indices, 
                                                        use_kmeans=use_kmeans, 
                                                        thd_noise=thd_noise,
                                                        classify_mu=classify_mu, 
                                                        sil_dist = sil_dist, 
                                                        thd_sil=thd_sil, 
                                                        thd_cent_dist=thd_cent_dist,
                                                        sc_tm=sc_tm, nc_tm=nc_tm)
            sil_scores = np.tile(cls_values["sil_scores"], (s2.shape[1], 1)).T
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr, cls_values_curr, _, _, _ = spike_classification(s2=s2, 
                                                             peak_indices=peak_indices, 
                                                             use_kmeans=use_kmeans, 
                                                             thd_noise=thd_noise,
                                                             classify_mu=classify_mu, 
                                                             sil_dist = sil_dist, 
                                                             thd_sil=thd_sil, 
                                                             thd_cent_dist=thd_cent_dist,
                                                             sc_tm=sc_tm, nc_tm=nc_tm)
            MUPulses_curr = MUPulses_curr + int(time*fs)

            sil_scores_curr = np.tile(cls_values_curr["sil_scores"], (s2.shape[1], 1)).T
            sil_scores_curr = batch_sil(sil_scores, 
                                        sil_scores_curr, 
                                        overlap=overlap, 
                                        fs=fs)
            sil_scores = sil_scores_curr
            
            for j in range(MUPulses.shape[0]):
                add_MUPulses = MUPulses_curr[j][MUPulses_curr[j] >= (time+overlap)*fs]
                tmp.append( np.array( np.append(MUPulses[j], add_MUPulses), dtype="int64" ) )
                
            tmp = np.array(tmp, dtype="object")
            MUPulses_curr = tmp
            MUPulses = MUPulses_curr    
            time += batch_size-overlap

        if time >= end_time:
            break

    decomp = {"MUPulses": MUPulses, "sil_scores": sil_scores}
    return decomp


def batch_decomp_window(data, B_realtime, mean_tm=None, discard=None, l=31,
                 use_kmeans=False, thd_noise=0.38,
                 classify_mu=True, sil_dist = True,
                 thd_sil=0.9, thd_cent_dist=0.6,
                 thd_pps=5,
                 sc_tm=None, nc_tm=None,
                 batch_size=0.6, overlap=0.3, fs=2048):
    if ((data[0][0].size == 0 or
         data[12][0].size == 0) and data.ndim == 2) or data.ndim == 3:
        end_time = data[0][1].shape[1] / fs
    else:
        end_time = data.shape[1] / fs
    time = 0.0
    while True:
        raw = crop_data(data, start = time, end = time+batch_size)

        # Source extraction
        s, _ = source_extraction(x = raw, 
                                 B_realtime = B_realtime, 
                                 mean_tm = mean_tm,
                                 discard=discard)

        # Peak extraction
        s2, peak_indices = peak_extraction(s=s, l=l)
        
        # Spike classification
        if time == 0.0:
            MUPulses, cls_values, _, _, _ = spike_classification(s2=s2, 
                                                        peak_indices=peak_indices, 
                                                        use_kmeans=use_kmeans, 
                                                        thd_noise=thd_noise,
                                                        classify_mu=classify_mu, 
                                                        sil_dist = sil_dist, 
                                                        thd_sil=thd_sil, 
                                                        thd_cent_dist=thd_cent_dist,
                                                        sc_tm=sc_tm, nc_tm=nc_tm)
            sil_scores = np.tile(cls_values["sil_scores"], (s2.shape[1], 1)).T
            thd_spikes = int(thd_pps * s2.shape[1]/fs)
                
            for i in range(MUPulses.shape[0]):
                # number of spikes in current window
                n_spikes_curr_i = MUPulses[i].shape[0]
                if (n_spikes_curr_i < thd_spikes):
                    MUPulses[i] = np.array([], dtype="int64")
                
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr, cls_values_curr, _, _, _ = spike_classification(s2=s2, 
                                                             peak_indices=peak_indices, 
                                                             use_kmeans=use_kmeans, 
                                                             thd_noise=thd_noise,
                                                             classify_mu=classify_mu, 
                                                             sil_dist = sil_dist, 
                                                             thd_sil=thd_sil, 
                                                             thd_cent_dist=thd_cent_dist,
                                                             sc_tm=sc_tm, nc_tm=nc_tm)
             
            MUPulses_curr = MUPulses_curr + int(time*fs)

            sil_scores_curr = np.tile(cls_values_curr["sil_scores"], (s2.shape[1], 1)).T
            sil_scores_curr = batch_sil(sil_scores, 
                                        sil_scores_curr, 
                                        overlap=overlap, 
                                        fs=fs)
            sil_scores = sil_scores_curr
            thd_spikes = int(thd_pps * s2.shape[1]/fs)

            for i in range(MUPulses.shape[0]):
                # number of spikes in current window
                n_spikes_curr_i = MUPulses_curr[i].shape[0]
                
                if (n_spikes_curr_i >= thd_spikes) and (n_spikes_curr_i >= thd_pps):
                    add_MUPulses = MUPulses_curr[i][MUPulses_curr[i] >= (time+overlap)*fs]
                else:
                    add_MUPulses = np.array([], dtype="int64")
                MUPulses_curr_i = np.append(MUPulses[i], add_MUPulses)
                tmp.append( np.array( MUPulses_curr_i, dtype="int64" ) ) 
            tmp = np.array(tmp, dtype="object")
            MUPulses_curr = tmp
            MUPulses = MUPulses_curr    
            time += batch_size-overlap

        if time >= end_time:
            break

    decomp = {"MUPulses": MUPulses, "sil_scores": sil_scores}
    return decomp


#############################################################################################################################
### kmeans_cp_sil  ##########################################################################################################
#############################################################################################################################
def sort_peaks2(s2, peak_indices, use_kmeans=True, random_seed=None, thd_noise=0.38):
    signal_clusters = []
    signal_centroids = []
    noise_clusters = []
    noise_centroids = []
    n_signal = []
    n_noise = []
    peak_indices_signal = []
    peak_indices_noise = []

    for i in range(s2.shape[0]):
        if use_kmeans:
            # Separating large peaks from relatively small peaks (noise)
            kmeans = KMeans(n_clusters=2, random_state=random_seed)
            kmeans.fit(s2[i][peak_indices[i]].reshape(-1,1))

            # Signal cluster centroid (sc_i)
            sc_i = np.argmax(kmeans.cluster_centers_)
            # Determining which cluster contains large peaks (signal)
            signal_indices = kmeans.labels_.astype(bool) # if sc_i == 1
            if sc_i == 0:
                signal_indices = ~signal_indices      
        else:
            # noise: peaks < thd_noise*s2.max()
            signal_indices = s2[i][peak_indices[i]] > thd_noise*s2[i].max()
            # signal_indices = signal_indices.flatten()
        n_signal_idx = signal_indices.sum()
        n_noise_idx = (~signal_indices).sum()
        
        # Indices of the peaks in signal cluster
        peak_indices_signal_i = peak_indices[i][signal_indices]
        peak_indices_noise_i = peak_indices[i][~signal_indices]
        peak_indices_signal.append(peak_indices_signal_i)
        peak_indices_noise.append(peak_indices_noise_i)

        # Signal cluster and Noise cluster
        signal_cluster = s2[i][peak_indices_signal_i]
        noise_cluster = np.delete(s2[i], peak_indices_signal_i)

        # Centroids
        signal_centroid = signal_cluster.mean()
        noise_centroid = noise_cluster.mean()
        
        signal_clusters.append(signal_cluster)
        signal_centroids.append(signal_centroid)
        noise_clusters.append(noise_cluster)
        noise_centroids.append(noise_centroid)
        n_signal.append(n_signal_idx)
        n_noise.append(n_noise_idx)

    n_signal = np.array(n_signal, dtype="int")
    n_noise = np.array(n_noise, dtype="int")
    peak_indices_signal = np.array(peak_indices_signal, dtype="object")
    peak_indices_noise = np.array(peak_indices_noise, dtype="object")
    
    signal_centroids = np.array(signal_centroids, dtype="float")
    signal_clusters = np.array(signal_clusters, dtype="object")
    
    noise_clusters = np.array(noise_clusters, dtype="object")
    noise_centroids = np.array(noise_centroids, dtype="float")
    
    # Distance between centroids
    centroid_dists = signal_centroids - noise_centroids 

    signal = {"n_signal": n_signal, 
            "peak_indices_signal": peak_indices_signal, 
            "signal_clusters": signal_clusters,
            "signal_centroids": signal_centroids}
    noise = {"n_noise": n_noise, 
            "peak_indices_noise": peak_indices_noise, 
            "noise_clusters": noise_clusters,
            "noise_centroids": noise_centroids}

    return signal, noise, centroid_dists, s2


def spike_classification2(s2, peak_indices,
                         use_kmeans=True, random_seed=None, thd_noise=0.38, 
                         classify_mu=True, sil_dist=False, 
                         thd_sil=0.9, thd_cent_dist=0.6,
                         sc_tm=None, nc_tm=None):
    """
    Returns a matrix of motor unit pulses.

    Args:
        s2              : numpy.ndarray
            Matrix containing squared source vectors
        peak_indices    : numpy.ndarray
            Matrix containing indices of detected peaks from s2
        use_kmeans      : bool
            Separates large peaks from small peaks using kmeans clustering if True
        random_seed     : int
            Used to initialize the pseudo-random processes in the function
        thd_sil         : float
            Threshold of the silhouette score
        thd_dist_ratio  : float
            Threshold of the peak distance ratio
        sil_dist        : bool
            Classifies peaks as motor unit pulses according to the silhouette score (if True) 
            or peak distance ratio (if False) 
    
    Returns:
        MUPulses        : numpy.ndarray
            Matrix containing indices of motor unit pulses
    """
    signal, noise, centroid_dists, s2, _ = sort_peaks(s2=s2, peak_indices=peak_indices, 
                                                           use_kmeans=use_kmeans, random_seed=random_seed, 
                                                           thd_noise=thd_noise)

    peak_indices_signal = signal["peak_indices_signal"]
    MUPulses = []
    sil_scores = []
    cent_dists = []
    for i in range(s2.shape[0]):
        add_peaks = peak_indices_signal[i]
        
        # Comparing distance of each peak to signal centroid (sc_tm) and noise centroid (nc_tm), 
        # adding peaks closer to signal centroid
        if (sc_tm is not None) and (nc_tm is not None) and (len(add_peaks) != 0):
            # s2_max = s2[i][add_peaks].max()
            # add_indices = ( abs(s2[i][add_peaks]/s2_max - nc_tm[i]) > 
            #                abs(s2[i][add_peaks]/s2_max - sc_tm[i]) )
            # add_peaks = add_peaks[add_indices] 
            add_indices = ( abs(s2[i][add_peaks] - (nc_tm[i]/sc_tm.max())) > 
                            abs(s2[i][add_peaks] - (sc_tm[i]/sc_tm.max())) )
            add_peaks = add_peaks[add_indices] 
        
        if classify_mu: # whether MU i is firing or not, based on SIL or centroid distance
            if sil_dist:
                # Silhouette score
                sil = silhouette_score(s2[i], peak_indices_signal[i])
                sil_scores.append(sil)
                if sil < thd_sil:
                    add_peaks = []
            else:
                cent_dist = centroid_dists[i] 
                cent_dists.append(cent_dist)
                if cent_dist < thd_cent_dist:
                    add_peaks = []
        
        MUPulses.append(np.array(add_peaks, dtype="int64"))
    
    length = len(MUPulses[0])
    if any(len(arr) != length for arr in MUPulses):
        MUPulses = np.array(MUPulses, dtype="object")
    else:
        MUPulses = np.array(MUPulses, dtype="int64")
        
    cls_values = {"centroid_dists": np.array(cent_dists, dtype="float"),
                  "sil_scores": np.array(sil_scores, dtype="float")}

    return MUPulses, cls_values, signal, noise, s2



def batch_decomp2(data, B_realtime, mean_tm=None, discard=None, l=31,
                 use_kmeans=False, thd_noise=0.38,
                 classify_mu=True, sil_dist = True,
                 thd_sil=0.9, thd_cent_dist=0.6,
                 sc_tm=None, nc_tm=None,
                 batch_size=4.0, overlap=0.0, fs=2048):
    if (data[0][0].size == 0 and data.ndim == 2) or data.ndim == 3:
        end_time = data[0][1].shape[1] / fs
    else:
        end_time = data.shape[1] / fs

    time = 0.0
    while True:
        raw = crop_data(data, start = time, end = time+batch_size)

        # Source extraction
        s, _ = source_extraction(x = raw, 
                                 B_realtime = B_realtime, 
                                 mean_tm = mean_tm,
                                 discard=discard)

        # Peak extraction
        s2, peak_indices = peak_extraction(s, l=l)
        
        # Spike classification
        if time == 0.0:
            MUPulses, _, _, _, _ = spike_classification2(s2=s2, 
                                                        peak_indices=peak_indices, 
                                                        use_kmeans=use_kmeans, 
                                                        thd_noise=thd_noise,
                                                        classify_mu=classify_mu, 
                                                        sil_dist = sil_dist, 
                                                        thd_sil=thd_sil, 
                                                        thd_cent_dist=thd_cent_dist,
                                                        sc_tm=sc_tm, nc_tm=nc_tm)
            time += batch_size-overlap
        else:
            tmp = []
            MUPulses_curr, _, _, _, _ = spike_classification2(s2=s2, 
                                                             peak_indices=peak_indices, 
                                                             use_kmeans=use_kmeans, 
                                                             thd_noise=thd_noise,
                                                             classify_mu=classify_mu, 
                                                             sil_dist = sil_dist, 
                                                             thd_sil=thd_sil, 
                                                             thd_cent_dist=thd_cent_dist,
                                                             sc_tm=sc_tm, nc_tm=nc_tm)
            MUPulses_curr = MUPulses_curr + int(time*fs)

            for j in range(MUPulses.shape[0]):
                add_MUPulses = MUPulses_curr[j][MUPulses_curr[j] >= (time+overlap)*fs]
                tmp.append( np.array( np.append(MUPulses[j], add_MUPulses), dtype="int64" ) )
                
            tmp = np.array(tmp, dtype="object")
            MUPulses_curr = tmp
            MUPulses = MUPulses_curr    
            time += batch_size-overlap

        if time >= end_time:
            break

    decomp = {"MUPulses": MUPulses}
    return decomp


#############################################################################################################################
### Plotting ################################################################################################################
#############################################################################################################################
def plot_extracted_peaks(s2, peak_indices, fs=2048, title="extracted peaks"):
    # Creating subplot
    n_rows = s2.shape[0]
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    time = np.arange(0, s2.shape[1], dtype="float") / float(fs)
    
    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=40)
    for i in range(s2.shape[0]):
        y = s2[i]
        ax[i].plot(time, y)
        ax[i].set_ylabel(f"MU {i}", fontsize=20)
        if len(peak_indices[i]) != 0:
            ax[i].scatter(peak_indices[i]/fs, s2[i][peak_indices[i]], c='r', s=40)
    plt.show()


def plot_classified_spikes(s2, peak_indices, MUPulses, fs=2048, 
                           title="classified spikes", label1="detected peaks", label2="MUPulses"):
    font_large = 24
    font_medium = 20
    font_small = 16
    
    # Creating subplot
    n_rows = s2.shape[0]
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    time = np.arange(0, s2.shape[1], dtype="float") / float(fs)
    
    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=font_large)
    for i in range(s2.shape[0]):
        ax[i].plot(time, s2[i])
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        if len(peak_indices[i]) != 0:
            ax[i].scatter(peak_indices[i]/fs, s2[i][peak_indices[i]], c='g', s=70, label=label1)
        if len(MUPulses[i]) != 0:
            ax[i].scatter(MUPulses[i]/fs, s2[i][MUPulses[i]], c='r', s=40, label=label2)
        if i == 0:
            ax[i].legend(loc='upper right', shadow=False, fontsize=font_medium)
    plt.show()


def plot_peaks(s2, noise, signal, centroid_dists, fs=2048, title="extracted peaks"):
    font_large = 30
    font_medium = 20
    font_small = 16
    
    # Creating subplot
    n_rows = s2.shape[0]
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    t_axis = np.arange(0, s2.shape[1], dtype="float") / float(fs)

    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=font_large)
    for i in range(s2.shape[0]):
        ax[i].plot(t_axis, s2[i], label=r"$s^2$")
        ax[i].set_ylabel(f"MU {i}", fontsize=font_medium)
        if noise["peak_indices_noise"][i].size != 0:
            ax[i].scatter(noise["peak_indices_noise"][i]/fs, s2[i][noise["peak_indices_noise"][i]], c='r', s=40, label="noise")
        ax[i].scatter(signal["peak_indices_signal"][i]/fs, s2[i][signal["peak_indices_signal"][i]], c='g', s=40, label="signal")
        ax[i].scatter(signal["peak_indices_signal"][i][signal["signal_clusters"][i].argmax()]/fs, signal["signal_clusters"][i].max(), c='c', s=80)
        ax[i].xaxis.set_tick_params(labelsize=font_small)
        ax[i].yaxis.set_tick_params(labelsize=font_small)
        ax[i].text(.02,.95, f"centroid_dist={centroid_dists[i]:.4f}", fontsize=font_medium, transform=ax[i].transAxes)
        ax[i].legend(loc='upper right', shadow=False, fontsize=font_medium)
    plt.show()


def plot_peaks_pulses(s2, noise, signal, sc_tm, nc_tm, MUPulses, fs=2048, title="extracted peaks"):
    font_large = 30
    font_medium = 20
    font_small = 16
    
    # Creating subplot
    n_rows = s2.shape[0] * 2
    height_ratio = np.ones(n_rows)
    plt.rcParams['figure.figsize'] = [35, 5*(n_rows)]
    fig, ax = plt.subplots(n_rows, 1, gridspec_kw={'height_ratios': height_ratio})
    t_axis = np.arange(0, s2.shape[1], dtype="float") / float(fs)

    # Plotting s2 and detected peaks
    ax[0].set_title(title, fontsize=font_large)
    for i in range(s2.shape[0]):
        ax[2*i].plot(t_axis, s2[i], label=r"$s^2$")
        ax[2*i].set_ylabel(f"MU {i}", fontsize=font_medium)
        if noise["peak_indices_noise"][i].size != 0:
            ax[2*i].scatter(noise["peak_indices_noise"][i]/fs, s2[i][noise["peak_indices_noise"][i]], c='r', s=40, label="noise")
        ax[2*i].scatter(signal["peak_indices_signal"][i]/fs, s2[i][signal["peak_indices_signal"][i]], c='g', s=40, label="signal")
        ax[2*i].scatter(signal["peak_indices_signal"][i][signal["signal_clusters"][i].argmax()]/fs, signal["signal_clusters"][i].max(), c='c', s=80)
        ax[2*i].xaxis.set_tick_params(labelsize=font_small)
        ax[2*i].yaxis.set_tick_params(labelsize=font_small)
        ax[2*i].text(.02,.95, f"sc={sc_tm[i]:.4f}\nnc={nc_tm[i]:.4f}", fontsize=font_medium, transform=ax[2*i].transAxes)
        ax[2*i].legend(loc='upper right', shadow=False, fontsize=font_medium)

        ax[2*i+1].plot(t_axis, s2[i])
        ax[2*i+1].set_ylabel(f"MU {i}", fontsize=20)
        ax[2*i+1].scatter(MUPulses[i]/fs, s2[i][MUPulses[i]], c='g', s=40, label="MUPulses")
        ax[2*i+1].legend(loc='upper right', shadow=False, fontsize=font_medium)
    plt.show()

