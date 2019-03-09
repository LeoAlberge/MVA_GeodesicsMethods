import numpy as np
from .nt_toolbox.grad import grad
from .nt_toolbox.signal import  bilinear_interpolate


def build_circle(n=100, r=2):
    gamma = []
    for i in range(n):
        gamma.append(np.cos(i*2*np.pi/n)+1j*np.sin(i*2*np.pi/n))
    return np.array(gamma)*r


def periodize(c):
    return np.concatenate((c, [c[0]]))


def interpc(x,xf,yf):
    return np.interp(x, xf, np.real(yf)) + 1j * np.interp(x, xf, np.imag(yf))


def curvabs(gamma):
    return np.concatenate(([0], np.cumsum( 1e-5 + abs(gamma[:-1:]-gamma[1::]) ) ) )


def resample1(gamma, d, p):
    return interpc(np.arange(0,p)/float(p),  d/d[-1], gamma)


def resample(gamma, p):
    return resample1(periodize(gamma), curvabs(periodize(gamma)), p)


def shiftR(c):
    return np.concatenate( ([c[-1]],c[:-1:]) )


def shiftL(c):
    return np.concatenate( (c[1::],[c[0]]) )


def BwdDiff(c):
    return c - shiftR(c)


def FwdDiff(c):
    return shiftL(c) - c


def normalize( v):
    return v/np.maximum(abs(v),1e-10)


def tangent( gamma):
    return normalize( FwdDiff(gamma) )


def normal( gamma):
    return -1j*tangent(gamma)


def normal_curvature(gamma):
    return BwdDiff(tangent(gamma)) / abs( FwdDiff(gamma) )


def compute_gradient(x):
    """
    Computes gradient in imaginary form.
    """
    res = grad(x)
    return res[:,:,0] + 1j*res[:,:,1]


def evaluate_curve( c, x):
    return bilinear_interpolate(x, np.imag(c), np.real(c))


def dot_product( c1, c2):
    return np.real(c1)*np.real(c2) + np.imag(c1)*np.imag(c2)


def conv_circ(signal, ker):
    """
    :param signal: real 1D array
    :param ker: real 1D array
    :return:  convolution
    """
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

