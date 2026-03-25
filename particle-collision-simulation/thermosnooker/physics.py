"""Physics module.

This module contains the Maxwell-Boltzmann distribution function to calculate the probability density function (PDF).

"""
import numpy as np

def maxwell(speed, kbt, mass):
    """Returns Maxwell-Boltzmann PDF value for a given temperature, speed and mass
    
    """
    sigma = np.sqrt(kbt / mass)
    prefactor = np.sqrt(2 / np.pi) * (speed**2) / (sigma**3)
    exponent = np.exp(- speed**2 / (2 * sigma**2))
    return prefactor * exponent
