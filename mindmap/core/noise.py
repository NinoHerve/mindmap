"""
Docstring pour mindmap.core.signals
"""

import numpy as np


def color_noise(t, n=1, filt=lambda f: 1):
    """
    Produces colored noise. Inspired from https://stackoverflow.com/questions/67085963/generate-colors-of-noise-in-python

    Parameters
    ----------
    t: int
        Number of samples per series.

    n: int
        Number of series.

    filt: function
        Frequency filter (depends on color)
        - f* = f (= white noise)
        - f* = 1/f (= pink noise)
        - ...

    Returns
    -------
    color: np.ndarray(n,1)
        Colored noise 

    Note: To produce colored noise we generate a white noise signal and fix the frequencies according to the filter.
    """
    color = np.zeros((n,t))

    for i in range(n):
        # generate white noise
        white     = np.random.randn(t)
        white_fft = np.fft.rfft(white)

        # fix the frequency space
        scale     = filt(np.fft.rfftfreq(t))
        scale     = scale / np.sqrt(np.mean(scale**2))      # preserve white noise energy
        color_fft = white_fft * scale

        # inverse Fourier
        color[i,:] = np.fft.irfft(color_fft)

    return color.squeeze()

def colored_noise(filt):
    """Applies noise with given filter"""
    return lambda t, n=1: color_noise(t, n, filt)

@colored_noise
def white_noise(_):
    """White noise filter"""
    return 1

@colored_noise
def blue_noise(f):
    """Blue noise filter"""
    return np.sqrt(f)

@colored_noise
def violet_noise(f):
    """Violet noise filter"""
    return f

@colored_noise
def brownian_noise(f):
    """Brownian noise filter"""
    return 1/np.where(f==0, float("inf"),f)

@colored_noise
def pink_noise(f):
    """Pink noise filter"""
    return 1/np.where(f==0, float("inf"), np.sqrt(f))

def get_noise_generator(color):
    """Utility to map string names to the functions above."""
    mapping = {
        "white": white_noise,
        "blue": blue_noise,
        "violet": violet_noise,
        "pink": pink_noise,
        "brown": brownian_noise,
        "brownian": brownian_noise,
    }
    if color not in mapping:
        raise ValueError(f"Noise color '{color}' not recognized.")
    return mapping[color]