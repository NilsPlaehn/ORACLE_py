import numpy as np
import numba

# JIT Compilation for Speed
@numba.njit
def S1_ORACLE_fct(bSSFPSignal, TR, alpha):
    NPhaseCycle = len(bSSFPSignal)
    phi         = np.linspace(0, 2 * np.pi, NPhaseCycle + 1)[:NPhaseCycle]

    VZ  = 1  # Right-handed coordinate system
    cm1 = np.conj(NPointFT(bSSFPSignal, -1 * VZ, phi))
    c0  = NPointFT(bSSFPSignal, 0 * VZ, phi)
    c1  = NPointFT(bSSFPSignal, 1 * VZ, phi)

    # ORACLE parameters
    z = c1 / c0
    x = np.abs(cm1) / np.abs(c0)
    r = np.abs(z)

    # treshold boundary conditions for T1 and T2 (not larger than 10s)
    valmax = np.exp(-0.005 / 10)

    # ORACLE solution functions
    E2     = np.minimum(np.abs((r + x) / (1 + x * r)), valmax)
    T2_est = -TR / np.log(E2)

    a      = E2 - 2 * r + E2 * r**2
    b      = E2 * (1 - 2 * E2 * r + r**2)
    E1     = np.minimum(np.abs((a + b * np.cos(alpha)) / (b + a * np.cos(alpha))), valmax)
    T1_est = -TR / np.log(E1)

    M0    = (np.abs(c0) + np.abs(cm1 / r)) / (2 * np.tan(alpha / 2)) * np.exp(TR / T2_est / 2)
    theta = np.angle(z)

    return T1_est, T2_est, theta, M0

@numba.njit
def NPointFT(MatInput, order, phi): # Discrete Fourier Transform maps bSSFP profile to bSSFP modes
    return np.sum(MatInput * np.exp(1j * order * phi)) / len(MatInput)

@numba.njit
def S1_bSSFP_Profile_Generation(M0, T1, T2, alpha, phi, TR, TE, theta):# bSSFP literature equation
    E1 = np.exp(-TR / T1)
    E2 = np.exp(-TR / T2)

    a = M0 * (1 - E1) * np.sin(alpha)
    b = 1 - E1 * E2**2 + (E2**2 - E1) * np.cos(alpha)
    c = 2 * (E1 - 1) * E2 * np.cos(alpha / 2)**2

    profile = (-1j * a / (b + c * np.cos(theta - phi)) *
               (1 - E2 * np.exp(-1j * (theta - phi))) *
               np.exp(-TE / T2) * np.exp(1j * theta * TE / TR))

    return profile
