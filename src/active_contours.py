import numpy as np
from numeric_tools import conv_circ, evaluate_curve, normal_curvature, dot_product,\
                           planar_curve, curvabs, compute_gradient, normal, resample,\
                            compute_region_term

from plotting_tools import show_fig_polar_curve, show_fig_standard_curve,show_fig_polar_curve_debug
import matplotlib.pyplot as plt


def K(s, gam, L):
    return (1 + ((s/L)**2 - (s/L) + 1/6)/(2*gam))/L


def kr(s, h, gam, L):
    return conv_circ(K(s, gam, L), h)


def gradient_standard(c, g, grad_g, N):
    return -evaluate_curve(c, g)*normal_curvature(c) + dot_product(evaluate_curve(c, grad_g), N) * N


def gradient_L2(L, c, N, g, grad_g, alpha, region_term=0):
    grad = region_term * N + alpha * (-evaluate_curve(c, g) * normal_curvature(c) + dot_product(evaluate_curve(c, grad_g), N) * N)
    return grad


def gradient_L2_new(c, N, **kwargs):

    if 'g' in kwargs and 'region_term' in kwargs:
        assert('alpha' in kwargs)
        assert('c1' in kwargs)
        assert('c2' in kwargs)
        grad = compute_region_term(c, kwargs['region_term'], kwargs['c1'], kwargs['c2'])*N + kwargs['alpha'] * (-evaluate_curve(c, kwargs['g']) * normal_curvature(c) + dot_product(evaluate_curve(c, kwargs['grad_g']), N) * N)

    elif 'g' in kwargs and 'region_term' not in kwargs:
        grad = -evaluate_curve(c, kwargs['g']) * normal_curvature(c) + dot_product(evaluate_curve(c, kwargs['grad_g']), N) * N

    elif 'g' not in kwargs and 'region_term' in kwargs:
        grad = compute_region_term(c, kwargs['region_term'], kwargs['c1'], kwargs['c2']) * N

    else:
        raise(AttributeError, 'No gradient term was given')
    return grad


def gradient_L(c, gradient_l2, L, N, lam, theta):
    ct_0 = gradient_l2.mean()
    ct_r = dot_product(np.cos(theta) + 1j*np.sin(theta), gradient_l2)/(lam)
    return ct_0, ct_r


def gradient_sobolev(c, gradient_l_r, gamma, L):
    ct_r = kr(c, gradient_l_r, gamma, L)
    return ct_r


def perform_gradient_descent_standard_curve(g,
                                            c_0,
                                            dt=1,
                                            niter=5000,
                                            nb_points_c=256,
                                            step_display=1000,
                                            region_term=0,
                                            alpha=1
                                            ):
    theta = np.transpose(np.linspace(0, 2 * np.pi, nb_points_c + 1))
    theta = theta[0:-1]

    fig = plt.figure(figsize=(4, 4))
    grad_g = compute_gradient(g)
    c = c_0.copy()
    c = resample(c, nb_points_c)

    for i in range(niter):
        N = normal(c)
        L = curvabs(c)[-1]
        grad_s = gradient_L2(L, c, N, g, grad_g, alpha, region_term)

        c = c - dt * grad_s
        c = resample(c, nb_points_c)
        if i % step_display == 0:
            show_fig_standard_curve(fig,
                                    g,
                                    c,
                                    grad_s,
                                    N,
                                    theta,
                                    show_grad=True,
                                    show_background=True,
                                    clear=True,
                                    timing=0.1)

    show_fig_standard_curve(fig,
                            g,
                            c,
                            grad_s,
                            N,
                            theta,
                            show_grad=True,
                            show_background=True,
                            clear=False,
                            timing=0.1)

# def perform_gradient_descent_polar_curve(g,
#                                          c_0,
#                                          c_r,
#                                          gamma,
#                                          alpha,
#                                          lam,
#                                          dt,
#                                          niter=50000,
#                                          nb_points_c=256,
#                                          step_display=1000,
#                                          region_term=0,
#                                          sobolev=True,
#                                          save=None,
#                                          t=1,
#                                          f=None,
#                                          c1=None,
#                                          c2=None
#                                          ):
#     fig = plt.figure(figsize=(10, 4))
#     plt.subplots_adjust(hspace=1)
#
#     theta = np.transpose(np.linspace(0, 2 * np.pi, nb_points_c + 1))
#     theta = theta[0:-1]
#
#     grad_g = compute_gradient(g)
#     #grad_g = rescale(np.minimum(grad_g, .05), .3, 1)
#
#     c = planar_curve(c_0, c_r, theta)
#
#     for i in range(niter):
#         c = planar_curve(c_0, c_r, theta)
#         c = resample(c, nb_points_c)
#         N = normal(c)
#         L = curvabs(c)[-1]
#         curvab = curvabs(c)
#
#         if f is not None:
#             region_term = compute_region_term(c, f, c1, c2)
#
#         grad_l2 = gradient_L2(L, c, N, g, grad_g, alpha, region_term)
#         ct_0, ct_rl = gradient_L(c, grad_l2, L, N, lam, theta)
#
#         if sobolev:
#             ct_r = gradient_sobolev(curvab, ct_rl, gamma, L)
#         else:
#             ct_r = ct_rl
#
#         c_0 = c_0 - dt * ct_0
#         c_r = c_r - dt * ct_r * t
#
#         if i % step_display == 0:
#             show_fig_polar_curve(fig=fig,
#                                  W=g,
#                                  c=c,
#                                  c_0=c_0,
#                                  c_r=c_r,
#                                  ct_0=ct_0,
#                                  ct_r=ct_r,
#                                  N=N,
#                                  theta=theta,
#                                  show_grad_cr=True,
#                                  show_grad_c0=True,
#                                  show_background=True,
#                                  clear=True,
#                                  timing=0.5)
#
#     show_fig_polar_curve(fig=fig,
#                          W=g,
#                          c=c,
#                          c_0=c_0,
#                          c_r=c_r,
#                          ct_0=ct_0,
#                          ct_r=ct_r,
#                          N=N,
#                          theta=theta,
#                          show_grad_cr=True,
#                          show_grad_c0=True,
#                          show_background=True,
#                          clear=True,
#                          timing=0.5,
#                          save=save)
#
#     return c_0, c_r


def perform_gradient_descent_polar_curve(I,
                                         c_0,
                                         c_r,
                                         dt,
                                         niter,
                                         nb_points_c,
                                         sobolev=True,
                                         lam=1,
                                         **kwargs
                                         ):
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=1)

    theta = np.transpose(np.linspace(0, 2 * np.pi, nb_points_c + 1))
    theta = theta[0:-1]

    if 'g' in kwargs:
        kwargs['grad_g'] = compute_gradient(kwargs['g'])

    c = planar_curve(c_0, c_r, theta)

    for i in range(niter):
        c = planar_curve(c_0, c_r, theta)
        c = resample(c, nb_points_c)
        N = normal(c)
        L = curvabs(c)[-1]
        curvab = curvabs(c)

        grad_l2 = gradient_L2_new(c, N, **kwargs)
        ct_0, ct_rl = gradient_L(c, grad_l2, L, N, lam, theta)

        if sobolev:
            assert('gamma' in kwargs)
            ct_r = gradient_sobolev(curvab, ct_rl, kwargs['gamma'], L)
        else:
            ct_r = ct_rl

        c_0 = c_0 - dt * ct_0
        c_r = c_r - dt * ct_r

        if 'step_display' in kwargs and  i % kwargs['step_display'] == 0:
            show_fig_polar_curve(fig=fig,
                                 W=I,
                                 c=c,
                                 c_0=c_0,
                                 c_r=c_r,
                                 ct_0=ct_0,
                                 ct_r=ct_r,
                                 N=N,
                                 theta=theta,
                                 show_grad_cr=True,
                                 show_grad_c0=True,
                                 show_background=True,
                                 clear=True,
                                 timing=0.5)

    show_fig_polar_curve(fig=fig,
                         W=I,
                         c=c,
                         c_0=c_0,
                         c_r=c_r,
                         ct_0=ct_0,
                         ct_r=ct_r,
                         N=N,
                         theta=theta,
                         show_grad_cr=True,
                         show_grad_c0=True,
                         show_background=True,
                         timing=0.5,
                         **kwargs)

    return c_0, c_r


def follow_object_on_frame():
    # TODO: Implementer le suivi d'un objet
    pass

