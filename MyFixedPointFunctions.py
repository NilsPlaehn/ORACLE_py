import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# ============================ Core Computation Functions ============================ #
@njit(fastmath=True)
def Fixed_Point_validation(T1, T2, nPC, percent, indvari, nvari, nPC_infty):
    """Performs fixed-point validation using optimized computations."""
    TR, TE, alpha, M0 = 5e-3, 2.5e-3, 15 * np.pi / 180, 1
    theta = 4 * np.pi / nvari * indvari

    T2 = min(T2, T1)  # Ensure T2 <= T1

   
    A_inf, B_inf, z_inf = GetbSSFPmodes(M0, T1, T2, alpha, TR, TE, theta, nPC_infty)
    c0, c1, cm1 = GetDFTmodes(M0, T1, T2, alpha, TR, TE, theta, nPC)

    A_fin, z_fin, B_fin = c0, c1 / c0, cm1
    xi_fin = GetXi(A_fin, B_fin, z_fin, nPC, c0, c1, cm1)

    Ab, Bb, zb, xib = Iterate_DTF2BSSFP_best(c0, cm1, c1, nPC, percent)
    
    return A_inf, B_inf, z_inf, A_fin, B_fin, z_fin, Ab, Bb, zb, xib, xi_fin


@njit(fastmath=True)
def Iterate_DTF2BSSFP_best(c0, cm1, c1, nPC, percent):
    """Performs aliasing correction using optimized fixed-point iterations."""
    rho, chi = c1 / c0, cm1 / c0
    ntmp = 10000  # Use a large buffer for iterations
    zvec = np.zeros(ntmp, dtype=np.complex128)
    zvec[0] = rho
    xi1 = np.full(ntmp, np.inf)
    xib, k = 1e4, 0

    while k < ntmp - 1:
        z = Getz(rho, chi, zvec[k], nPC)
        zvec[k + 1] = z

        r = np.abs(z)
        denom = (1 - r ** (2 * nPC - 2))
        A = (c0 - cm1 * np.conj(z) ** (nPC - 1)) / denom
        B = (cm1 - c0 * z ** (nPC - 1)) / denom

        xitmp = GetXi(A, B, z, nPC, c0, c1, cm1)
        xi1[k] = xitmp

        if xitmp < xib:
            xib, Ab, Bb, zb = xitmp, A, B, z

        if xitmp < percent:
            break
        k += 1

    return Ab * (1 - zb ** nPC), Bb * (1 - np.conj(zb) ** nPC), zb, xib

@njit(fastmath=True)
def GetXi(A0, B0, z0, nPC, c0, c1, cm1):
    """Computes the RMSE metric Xi using optimized vectorized operations."""
    num = (
        (np.abs(A0 + B0 * np.conj(z0) ** (nPC - 1) - c0) ** 2) +
        (np.abs(A0 * z0 + B0 * np.conj(z0) ** (nPC - 2) - c1) ** 2) +
        (np.abs(A0 * z0 ** (nPC - 1) + B0 - cm1) ** 2)
    )
    denom = np.abs(c0) ** 2 + np.abs(cm1) ** 2 + np.abs(c1) ** 2

    return np.sqrt(num / denom) * 100

@njit(fastmath=True)
def NPointFT(MatInput, order, phi):
    """Computes N-point Fourier Transform using vectorized operations."""
    return np.sum(MatInput * np.exp(1j * order * phi)) / len(MatInput)

@njit(fastmath=True)
def Getz(rho, chi, z, n):
    """Computes the updated z value using optimized operations."""
    z_n = z ** (n - 1)
    denom = (1 - np.abs(z) ** (2 * n - 2))
    return rho - (chi - z_n) / denom * (1 - np.abs(z) ** 2) * np.conj(z) ** (n - 2)

@njit(fastmath=True)
def S_bSSFP_profile(M0, T1, T2, alpha, phi, TR, TE, theta):
    """Computes the bSSFP profile equation in a vectorized manner."""
    E1, E2 = np.exp(-TR / T1), np.exp(-TR / T2)
    a = M0 * (1 - E1) * np.sin(alpha)
    b = 1 - E1 * E2**2 + (E2**2 - E1) * np.cos(alpha)
    c = 2 * (E1 - 1) * E2 * np.cos(alpha / 2) ** 2

    return (-1j * a / (b + c * np.cos(theta - phi)) *
            (1 - E2 * np.exp(-1j * (theta - phi))) *
            np.exp(-TE / T2) * np.exp(1j * theta * TE / TR))

@njit(fastmath=True)
def GetDFTmodes(M0, T1, T2, alpha, TR, TE, theta, nPC):
    """Computes Discrete Fourier Transform (DFT) modes using vectorized operations."""
    phi = np.arange(nPC) * (2 * np.pi / nPC) 
    profile = S_bSSFP_profile(M0, T1, T2, alpha, phi, TR, TE, theta)

    return NPointFT(profile, 0, phi), NPointFT(profile, 1, phi), NPointFT(profile, -1, phi)

@njit(fastmath=True)
def GetbSSFPmodes(M0, T1, T2, alpha, TR, TE, theta, nPC):
    """Computes steady-state bSSFP modes using vectorized operations."""
    phi = np.arange(nPC) * (2 * np.pi / nPC)  
    profile = S_bSSFP_profile(M0, T1, T2, alpha, phi, TR, TE, theta)

    c0, c1, cm1 = NPointFT(profile, 0, phi), NPointFT(profile, 1, phi), NPointFT(profile, -1, phi)
    return c0, cm1, c1 / c0




