# cython: boundscheck=False, wraparound=False, cdivision=True
"""
fmm_near_cy.pyx — Cython near-field block computation for 2D Helmholtz BIE.

Build:
    pip install cython numpy
    python setup_fmm.py build_ext --inplace

Or one-liner:
    cythonize -i fmm_near_cy.pyx
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, atan2, cos, sin, M_PI, j0, y0, j1, y1

np.import_array()

ctypedef np.float64_t DTYPE_f
ctypedef np.complex128_t DTYPE_c


# 8-point Gauss-Legendre on [0,1]
cdef int NQ8 = 8
cdef double GT8[8]
cdef double GW8[8]

GT8[0] = 0.01985507175123188; GT8[1] = 0.10166676129318691
GT8[2] = 0.23723379504183550; GT8[3] = 0.40828267875217510
GT8[4] = 0.59171732124782490; GT8[5] = 0.76276620495816450
GT8[6] = 0.89833323870681309; GT8[7] = 0.98014492824876812

GW8[0] = 0.05061426814518813; GW8[1] = 0.11119051722668724
GW8[2] = 0.15685332293894344; GW8[3] = 0.18134189168918100
GW8[4] = 0.18134189168918100; GW8[5] = 0.15685332293894344
GW8[6] = 0.11119051722668724; GW8[7] = 0.05061426814518813


cdef void _compute_block(
    int nq, const double *qt, const double *qw,
    double obs_p0x, double obs_p0y, double obs_sx, double obs_sy,
    double obs_nx, double obs_ny, double obs_len,
    double src_p0x, double src_p0y, double src_sx, double src_sy,
    double src_nx, double src_ny, double src_len,
    double k, int obs_nd,
    double *s_re, double *s_im, double *k_re, double *k_im) noexcept nogil:

    cdef int qi, qj, a, b, idx
    cdef double to, wo, ts, ws, rx, ry, sx, sy
    cdef double dx, dy, dist, kr
    cdef double J0v, Y0v, J1v, Y1v
    cdef double g_re_val, g_im_val, h1_re, h1_im
    cdef double dk_re_val, dk_im_val, proj, w, c
    cdef double po[2]
    cdef double ps[2]

    for idx in range(4):
        s_re[idx] = 0.0; s_im[idx] = 0.0
        k_re[idx] = 0.0; k_im[idx] = 0.0

    for qi in range(nq):
        to = qt[qi]; wo = qw[qi]
        po[0] = 1.0 - to; po[1] = to
        rx = obs_p0x + to * obs_sx
        ry = obs_p0y + to * obs_sy

        for qj in range(nq):
            ts = qt[qj]; ws = qw[qj]
            ps[0] = 1.0 - ts; ps[1] = ts
            sx = src_p0x + ts * src_sx
            sy = src_p0y + ts * src_sy

            dx = rx - sx; dy = ry - sy
            dist = sqrt(dx*dx + dy*dy)
            if dist < 1e-15:
                dist = 1e-15
            kr = k * dist

            J0v = j0(kr); Y0v = y0(kr)
            J1v = j1(kr); Y1v = y1(kr)

            # G = (j/4) H0^(2) = (j/4)(J0 - jY0) = Y0/4 + j*J0/4
            g_re_val = 0.25 * Y0v
            g_im_val = 0.25 * J0v

            h1_re = J1v; h1_im = -Y1v

            if obs_nd:
                proj = (dx*obs_nx + dy*obs_ny) / dist
                dk_re_val = ( 0.25*k*h1_im) * proj
                dk_im_val = (-0.25*k*h1_re) * proj
            else:
                proj = (src_nx*dx + src_ny*dy) / dist
                dk_re_val = (-0.25*k*h1_im) * proj
                dk_im_val = ( 0.25*k*h1_re) * proj

            w = wo * ws * obs_len * src_len
            for a in range(2):
                for b in range(2):
                    c = po[a] * ps[b] * w
                    idx = a*2 + b
                    s_re[idx] += c * g_re_val
                    s_im[idx] += c * g_im_val
                    k_re[idx] += c * dk_re_val
                    k_im[idx] += c * dk_im_val


def compute_sk_blocks_batch(
    np.ndarray[DTYPE_f, ndim=1] qt_arr,
    np.ndarray[DTYPE_f, ndim=1] qw_arr,
    np.ndarray[DTYPE_f, ndim=2] obs_p0,
    np.ndarray[DTYPE_f, ndim=2] obs_seg,
    np.ndarray[DTYPE_f, ndim=2] obs_n,
    np.ndarray[DTYPE_f, ndim=1] obs_len,
    np.ndarray[DTYPE_f, ndim=2] src_p0,
    np.ndarray[DTYPE_f, ndim=2] src_seg,
    np.ndarray[DTYPE_f, ndim=2] src_n,
    np.ndarray[DTYPE_f, ndim=1] src_len,
    double k,
    int obs_nd):
    """
    Batch compute 2x2 SLP and K'/K blocks for P element pairs.

    Parameters
    ----------
    qt_arr, qw_arr : quadrature nodes/weights on [0,1]
    obs_p0, obs_seg, obs_n : (P,2) observer element geometry
    obs_len : (P,) observer lengths
    src_p0, src_seg, src_n : (P,2) source element geometry
    src_len : (P,) source lengths
    k : real wavenumber
    obs_nd : 0 for K (dn_src), 1 for K' (dn_obs)

    Returns
    -------
    s_blocks : (P,2,2) complex SLP blocks
    k_blocks : (P,2,2) complex K'/K blocks
    """
    cdef int P = obs_p0.shape[0]
    cdef int nq = qt_arr.shape[0]
    cdef int p

    cdef np.ndarray[DTYPE_f, ndim=1] s_re_flat = np.zeros(4*P, dtype=np.float64)
    cdef np.ndarray[DTYPE_f, ndim=1] s_im_flat = np.zeros(4*P, dtype=np.float64)
    cdef np.ndarray[DTYPE_f, ndim=1] k_re_flat = np.zeros(4*P, dtype=np.float64)
    cdef np.ndarray[DTYPE_f, ndim=1] k_im_flat = np.zeros(4*P, dtype=np.float64)

    cdef double *qt_ptr = <double*>qt_arr.data
    cdef double *qw_ptr = <double*>qw_arr.data

    for p in range(P):
        _compute_block(
            nq, qt_ptr, qw_ptr,
            obs_p0[p, 0], obs_p0[p, 1], obs_seg[p, 0], obs_seg[p, 1],
            obs_n[p, 0], obs_n[p, 1], obs_len[p],
            src_p0[p, 0], src_p0[p, 1], src_seg[p, 0], src_seg[p, 1],
            src_n[p, 0], src_n[p, 1], src_len[p],
            k, obs_nd,
            &s_re_flat[4*p], &s_im_flat[4*p],
            &k_re_flat[4*p], &k_im_flat[4*p])

    s_blocks = (s_re_flat + 1j * s_im_flat).reshape(P, 2, 2)
    k_blocks = (k_re_flat + 1j * k_im_flat).reshape(P, 2, 2)
    return s_blocks, k_blocks
