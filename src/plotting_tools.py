import matplotlib.pyplot as plt
import  numpy as np
from .numeric_tools import periodize


def cplot(c,s='b',lw=1):
    plt.plot(np.real(periodize(c)), np.imag(periodize(c)), s, linewidth=lw)
    plt.axis('equal')
    plt.axis('off')