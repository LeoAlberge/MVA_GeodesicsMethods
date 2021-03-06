import time

import matplotlib.pyplot as plt
import numpy as np
from nt_toolbox.signal import imageplot
from numeric_tools import periodize


def cplot(c, s='b', lw=1):
    plt.plot(np.real(periodize(c)), np.imag(periodize(c)), s, linewidth=lw)
    plt.axis('equal')
    plt.axis('off')


def format_func(value, t):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


def show_fig_polar_curve(fig, c, W,  **kwargs):
    """

    :param fig:
    :param c:
    :param W:
    :param kwargs:
    :return:
    """
    if 'c_r' in kwargs:
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title('Composante radiale du contour')
        ax1.plot(kwargs['theta'], -kwargs['c_r'])
        # plt.axhline(y=0, color='r', linestyle='-')
        ax1.set_xlabel(r"$ \theta $")
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    if 'c_r' in kwargs:
        ax3 = fig.add_subplot(2, 2, 3)
        fft = np.fft.fft(kwargs['c_r'])
        ax3.plot(np.abs(fft)[:int(len(fft) / 2)])
        ax3.set_yscale('log')
        ax3.set_title('Coefficients de Fourier de la componsante radiale  \n du contour')
        ax3.set_xlabel(r"$\omega$")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Gradients le long du contour')
    cplot(c)

    if 'c_0' in kwargs:
        plt.scatter(np.real(kwargs['c_0']), np.imag(kwargs['c_0']))

    show_grad_c0 = True if 'show_grad_c0' in kwargs else False
    show_grad_cr = True if 'show_grad_cr' in kwargs else False
    show_background = True if 'show_background' in kwargs else False

    if show_grad_c0 and 'c_0' in kwargs and 'ct_0' in kwargs:
        plt.quiver(np.real(kwargs['c_0']), np.imag(kwargs['c_0']), -np.real(kwargs['ct_0']), np.imag(kwargs['ct_0']), label=r"gradient $c_o$", alpha=0.5,
                   color='green')
    if show_grad_cr and 'ct_r' in kwargs and 'N' in kwargs:
        plt.quiver(np.real(c), np.imag(c), -kwargs['ct_r'] * np.real(kwargs['N']), kwargs['ct_r'] * np.imag(kwargs['N']), label=r"gradient $c_r$", alpha=0.5,
                   color='blue')

    if show_background:
        imageplot(np.transpose(W))
    plt.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Contour')
    cplot(c)
    if show_background:
        imageplot(np.transpose(W))

    fig.canvas.draw()  # draw
    clear = True if 'clear' in kwargs else False
    save = True if 'save' in kwargs else False
    save_contour = True if 'save_contour' in kwargs else False

    if save_contour:
        extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('{}.eps'.format(kwargs['save_contour']), bbox_inches=extent.expanded(1.3, 1.3))

    if save:
        if 'iter' in kwargs:
            plt.savefig('{}_{}.eps'.format(kwargs['save'], str(kwargs['iter'])), quality=100)

        else:
            plt.savefig('{}.eps'.format(kwargs['save']), quality=100)

    if clear and 'timing' in kwargs:
        time.sleep(kwargs['timing'])  # sleep
        fig.show()
        fig.clear()
    else:
        fig.show()


# def show_fig_polar_curve(fig,
#                           W,
#                           c,
#                           c_0,
#                           c_r,
#                           ct_0,
#                           ct_r,
#                           N,
#                           theta,
#                           show_grad_cr=True,
#                           show_grad_c0=True,
#                           show_fft=True,
#                           show_background=True,
#                           clear=True,
#                           timing=0.1,
#                           save=None):
#     ax1 = fig.add_subplot(2, 2, 1)
#     #     plt.title('L: {0:0.2f}, Mean ct_r: {1:0.2e}, ct_0: {2:0.2e},{3:0.2e}'.format(curvabs(c)[-1],
#     #                                                          ct_r.mean(),
#     #                                                          np.real(ct_0),
#     #                                                          np.imag(ct_0)
#     #                                                                      ))
#     ax1.set_title('Composante radiale')
#     ax1.plot(theta, -ct_r)
#     plt.axhline(y=0, color='r', linestyle='-')
#
#     ax1.set_xlabel(r"$ \theta $")
#     ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
#
#     ax3 = fig.add_subplot(2, 2, 3)
#     fft = np.fft.fft(c_r)
#     ax3.plot(np.abs(fft)[:int(len(fft) / 2)])
#     ax3.set_yscale('log')
#     ax3.set_title('Coefficients de Fourier')
#     ax3.set_xlabel(r"$\omega$")
#
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax2.set_title('Gradients')
#     cplot(c)
#     plt.scatter(np.real(c_0), np.imag(c_0))
#     if show_grad_c0:
#         plt.quiver(np.real(c_0), np.imag(c_0), -np.real(ct_0), np.imag(ct_0), label=r"gradient $c_o$", alpha=0.5,
#                    color='green')
#     if show_grad_cr:
#         plt.quiver(np.real(c), np.imag(c), -ct_r*np.real(N), ct_r*np.imag(N), label=r"gradient $c_r$", alpha=0.5,
#                    color='blue')
#
#     if show_background:
#         imageplot(np.transpose(W))
#     plt.legend()
#
#     ax4 = fig.add_subplot(2, 2, 4)
#     ax4.set_title('Curve')
#     cplot(c)
#     if show_background:
#         imageplot(np.transpose(W))
#
#     fig.canvas.draw()  # draw
#     if clear:
#         time.sleep(timing)  # sleep
#     fig.show()
#     if clear:
#         fig.clear()
#     if save:
#         plt.savefig(save)
#
#
# def show_fig_polar_curve_debug(fig,
#                                 W,
#                                 c,
#                                 c_0,
#                                 c_r,
#                                 ct_0,
#                                 ct_r,
#                                 ct_rl,
#                                 N,
#                                 theta,
#                                 show_grad_cr=True,
#                                 show_grad_c0=True,
#                                 show_fft=True,
#                                 show_background=True,
#                                 clear=True,
#                                 timing=0.1,
#                                 save=None):
#     ax1 = fig.add_subplot(2, 2, 1)
#     #     plt.title('L: {0:0.2f}, Mean ct_r: {1:0.2e}, ct_0: {2:0.2e},{3:0.2e}'.format(curvabs(c)[-1],
#     #                                                          ct_r.mean(),
#     #                                                          np.real(ct_0),
#     #                                                          np.imag(ct_0)
#     #                                                                      ))
#     ax1.set_title('Composante radiale - Gradient')
#     ax1.plot(theta, -ct_r)
#     plt.axhline(y=0, color='r', linestyle='-')
#
#     ax1.set_xlabel(r"$ \theta $")
#     ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
#
#
#     ax3 = fig.add_subplot(2, 2, 3)
#     ax3.set_title('Composante radiale - Gradient non smooth')
#     ax3.plot(theta, -ct_rl)
#     plt.axhline(y=0, color='r', linestyle='-')
#     ax3.set_xlabel(r"$ \theta $")
#     ax3.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
#
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax2.set_title('Gradients')
#     cplot(c)
#     plt.scatter(np.real(c_0), np.imag(c_0))
#     if show_grad_c0:
#         plt.quiver(np.real(c_0), np.imag(c_0), -np.real(ct_0), np.imag(ct_0), label=r"gradient $c_o$", alpha=0.5,
#                    color='green')
#     if show_grad_cr:
#         plt.quiver(np.real(c), np.imag(c), -ct_r*np.real(N), ct_r*np.imag(N), label=r"gradient $c_r$", alpha=0.5,
#                    color='blue')
#
#     if show_background:
#         imageplot(np.transpose(W))
#     plt.legend()
#
#     ax4 = fig.add_subplot(2, 2, 4)
#     ax4.set_title('Curve')
#     cplot(c)
#     if show_background:
#         imageplot(np.transpose(W))
#
#     fig.canvas.draw()  # draw
#     if clear:
#         time.sleep(timing)  # sleep
#     fig.show()
#     if clear:
#         fig.clear()
#     if save:
#         plt.savefig(save)

def show_fig_standard_curve(fig,
                            W,
                            c,
                            grad_g,
                            N,
                            theta,
                            show_grad=True,
                            show_background=True,
                            clear=True,
                            timing=0.1):
    ax = fig.add_subplot(111)
    cplot(c)
    if show_grad:
        plt.quiver(np.real(c), np.imag(c), -np.real(grad_g), -np.imag(grad_g), color='blue', alpha=0.5)
    if show_background:
        imageplot(np.transpose(W))
    fig.canvas.draw()  # draw
    if clear:
        time.sleep(timing)  # sleep
    fig.show()
    if clear:
        fig.clear()
