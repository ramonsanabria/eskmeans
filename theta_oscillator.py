"""
Oscillator-based speech syllabification algorithm (Räsänen et al., 2015, 2018).

Based on the papers:

- O. J. Räsänen, G. Doyle, and M. C. Frank, "Unsupervised word discovery from
  speech using automatic segmentation into syllable-like units," in Proc.
  Interspeech, 2015.
- O. J. Räsänen, G. Doyle, and M. C. Frank, "Pre-linguistic segmentation of
  speech into syllable-like units," Cognition, 2018.

This is a derivation of Adriana Stan's Python implementation available at
https://github.com/speech-utcluj/thetaOscillator-syllable-segmentation.
Modified by Herman Kamper (kamperh@gmail.com, 2021) and subsequently included
here.
"""

from pathlib import Path
from scipy.signal import hilbert
import gammatone.filters
import librosa
import numpy as np
import sys


def peakdet(v, delta, x=None):
    """Converted from MATLAB script at http://billauer.co.il/peakdet.html."""
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit("Input vectors v and x must have same length")
    if not np.isscalar(delta):
        sys.exit("Input argument delta must be a scalar")
    if delta <= 0:
        sys.exit("Input argument delta must be positive")

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan
    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx[0]))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn[0]))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def theta_oscillator(ENVELOPE, f=5, Q=0.5, thr=0.025, verbose=False):
    """Based on https://github.com/orasanen/thetaOscillator."""

    N = 8  # how many most energetic bands to use

    if N > ENVELOPE.size:
        print("WARNING: Input dimensionality smaller than N. Using all bands.")

    # Delay compensation lookup table
    a = np.array([
        [72,  34, 22, 16, 12,  9,  8,  6,  5,  4,  3,  3,  2,  2,  1,  0,  0,  0,  0,  0],
        [107, 52, 34, 25, 19, 16, 13, 11, 10,  9,  8,  7,  6,  5,  5,  4,  4,  4,  3,  3],
        [129, 64, 42, 31, 24, 20, 17, 14, 13, 11, 10,  9,  8,  7,  7,  6,  6,  5,  5,  4],
        [145, 72, 47, 35, 28, 23, 19, 17, 15, 13, 12, 10,  9,  9,  8,  7,  7,  6,  6,  5],
        [157, 78, 51, 38, 30, 25, 21, 18, 16, 14, 13, 12, 11, 10,  9,  8,  8,  7,  7,  6],
        [167, 83, 55, 41, 32, 27, 23, 19, 17, 15, 14, 12, 11, 10, 10,  9,  8,  8,  7,  7],
        [175, 87, 57, 43, 34, 28, 24, 21, 18, 16, 15, 13, 12, 11, 10,  9,  9,  8,  8,  7],
        [181, 90, 59, 44, 35, 29, 25, 21, 19, 17, 15, 14, 13, 12, 11, 10,  9,  9,  8,  8],
        [187, 93, 61, 46, 36, 30, 25, 22, 19, 17, 16, 14, 13, 12, 11, 10, 10,  9,  8,  8],
        [191, 95, 63, 47, 37, 31, 26, 23, 20, 18, 16, 15, 13, 12, 11, 11, 10,  9,  9,  8],
    ])

    i1 = max(0, min(10, round(Q * 10)))
    i2 = max(0, min(20, round(f)))
    delay_compensation = a[i1 - 1][i2 - 1]

    T = 1.0 / f
    k = 1
    b = 2 * np.pi / T
    m = k / b ** 2
    c = np.sqrt(m * k) / Q

    if verbose:
        print(f"Q={Q:.4f}, f={1/T:.1f} Hz, bw={1/T/Q:.1f} Hz")

    e = np.transpose(ENVELOPE)
    e = np.vstack((e, np.zeros((500, e.shape[1]))))
    F = e.shape[1]

    x = np.zeros((e.shape[0], F))
    a_arr = np.zeros((e.shape[0], F))
    v = np.zeros((e.shape[0], F))

    for t in range(1, e.shape[0]):
        for cf in range(F):
            f_up = e[t, cf]
            f_down = -k * x[t - 1, cf] - c * v[t - 1, cf]
            f_tot = f_up + f_down
            a_arr[t, cf] = f_tot / m
            v[t, cf] = v[t - 1, cf] + a_arr[t, cf] * 0.001
            x[t, cf] = x[t - 1, cf] + v[t, cf] * 0.001

    for fi in range(F):
        if delay_compensation:
            x[:, fi] = np.append(
                x[delay_compensation:, fi],
                np.zeros((delay_compensation, 1))
            )

    x = x[:-500]

    tmp = x - np.min(x) + 0.00001
    x = np.zeros((tmp.shape[0], 1))
    for zz in range(tmp.shape[0]):
        sort_tmp = np.sort(tmp[zz, :], axis=0)[::-1]
        x[zz] = sum(np.log10(sort_tmp[:N]))

    x = x - np.min(x)
    x = x / np.max(x)
    return x


def get_boundaries(wav_input, fs=None):
    """
    Compute syllable-like boundary times (in seconds) for an audio segment.

    Parameters
    ----------
    wav_input : str, Path, or np.ndarray
        Audio file path or raw samples (supply fs when passing samples).
    fs : int, optional
        Sample rate (required when wav_input is an array).

    Returns
    -------
    np.ndarray
        Boundary times in seconds, relative to the start of wav_input.
    """
    minfreq, maxfreq, bands = 50, 7500, 20

    cfs = np.zeros((bands, 1))
    const = (maxfreq / minfreq) ** (1 / (bands - 1))
    cfs[0] = 50
    for k in range(bands - 1):
        cfs[k + 1] = cfs[k] * const

    if isinstance(wav_input, (str, Path)):
        wav_data, fs = librosa.load(wav_input)
    else:
        wav_data = wav_input

    wav_data = librosa.resample(wav_data, orig_sr=fs, target_sr=16000)
    fs = 16000

    coefs = gammatone.filters.make_erb_filters(fs, cfs, width=1.0)
    filtered = gammatone.filters.erb_filterbank(wav_data, coefs)
    env = librosa.resample(np.abs(hilbert(filtered)), orig_sr=fs, target_sr=1000)

    outh = theta_oscillator(env, f=5, Q=0.5, thr=0.01)

    peaks, valleys = peakdet(outh, 0.01)

    if len(valleys) and len(peaks):
        valley_indices = valleys[:, 0]
        if valley_indices[0] > 50:
            valley_indices = np.insert(valley_indices, 0, 0)
        if valley_indices[-1] < env.shape[1] - 50:
            valley_indices = np.append(valley_indices, env.shape[1])
    else:
        valley_indices = np.array([0, env.shape[1]])

    return valley_indices / 1000.0
