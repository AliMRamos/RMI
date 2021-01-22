import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


def read_nist(file_data):
    """Read NIST data file and returns energy in MeV and Stopping Power in MeV*cm^2/g"""
    data = np.loadtxt(file_data, usecols=tuple(np.arange(0, 2)), skiprows=8).T
    return data[0], data[1]


def distance(file_data, ini_val, step_val):
    """Calculates the distance of an e- for a given initial energy and a fixed energy lost. The stopping power data are
    from the NIST"""
    data = read_nist(file_data)
    delta_x = []
    initial_energy = ini_val
    while round(initial_energy, 2) > 0.3:
        final_energy = initial_energy - step_val
        delta_x.append(
            quad(interp1d(data[0], 1 / data[1], kind='cubic'), final_energy, initial_energy)[0])
        initial_energy = final_energy
    return np.array(delta_x)


def lorentz_gamma_beta(energy, mass):
    """Calculates γ and β for a given energy and mass"""
    gamma = energy / mass + 1
    beta = np.sqrt(1 - (1 / gamma) ** 2)
    return gamma, beta


def momentum(energy, mass):
    """Obtains the momentum for a given energy and mass"""
    gamma, beta = lorentz_gamma_beta(energy, mass)
    p = mass * beta * gamma
    return p


def projected_angular_distribution(charge_number=1, x=None, x_0=None, beta=1, p=None):
    """Calculates θ_0 (projected angular distribution) for a given Δs (displacement)"""
    return 13.6 * charge_number * np.sqrt(x / x_0) * (
            1 + 0.038 * np.log(x * charge_number ** 2 / x_0)) / (beta * p)
