import matplotlib.pyplot as plt
import numpy as np
import time
from import_noise import *


def power_output_transmitted_fabry_perot_old(delta_L):
    """Taken from the Book a guide to experiments in optics, page 124"""
    loss_coefficient = 0.00000
    optical_path_length = 2 * (length_interferometer + delta_L)
    delta_phi = 2 * np.pi * (optical_path_length / wavelength - 1)

    output = amplitude_transmittance_mirror1 * amplitude_transmittance_mirror1 * np.exp(
        - loss_coefficient * (optical_path_length + delta_L) + 1j * delta_phi) / (
                     np.abs(1 - np.sqrt(amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2 * np.exp(
                         -loss_coefficient * optical_path_length)) * np.exp(-1j * delta_phi)) ** 2)
    return np.real(output)


def power_output_reflected_fabry_perot_old(delta_L):
    loss_coefficient = 0.00000
    optical_path_length = 2 * (length_interferometer + delta_L)
    delta_phi = 2 * np.pi * (optical_path_length / wavelength - 1)

    output = (amplitude_reflectance_mirror1 - (
            amplitude_reflectance_mirror1 + amplitude_transmittance_mirror1) * np.sqrt(
        amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2) * (
                      np.exp(-1j * delta_phi) + np.exp(1j * delta_phi)) + amplitude_reflectance_mirror2 * (
                      amplitude_reflectance_mirror1 + amplitude_transmittance_mirror1) ** 2) / (
                     np.abs(1 - np.sqrt(amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2 * np.exp(
                         -loss_coefficient * optical_path_length)) * np.exp(-1j * delta_phi)) ** 2)

    return np.real(output)


# the lower powers are the correct ones

def power_output_transmitted_fabry_perot(delta_L):
    loss_coefficient = 0.00000
    optical_path_length = 2 * (length_interferometer + delta_L)
    delta_phi = 2 * np.pi * (optical_path_length / wavelength - 1)

    output = amplitude_transmittance_mirror1 * amplitude_transmittance_mirror2 * np.exp(
        -loss_coefficient * optical_path_length) / (1 - 2 * np.sqrt(
        amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2 * np.exp(
            -loss_coefficient * optical_path_length)) * np.cos(
        delta_phi) + amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2 * np.exp(
        -loss_coefficient * optical_path_length))

    return output


def power_output_reflected_fabry_perot_old(delta_L):
    loss_coefficient = 0.00000
    optical_path_length = 2 * (length_interferometer + delta_L)
    delta_phi = 2 * np.pi * (optical_path_length / wavelength - 1)

    output = (amplitude_reflectance_mirror1 - 2 * np.sqrt(
        amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2) * (
                      amplitude_reflectance_mirror1 + amplitude_transmittance_mirror1) * np.cos(
        delta_phi) + amplitude_reflectance_mirror2 * (
                      amplitude_reflectance_mirror1 + amplitude_transmittance_mirror1) ** 2) / (
                     1 - 2 * np.sqrt(
                 amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2 * np.exp(
                     -loss_coefficient * optical_path_length)) * np.cos(
                 delta_phi) + amplitude_reflectance_mirror1 * amplitude_reflectance_mirror2 * np.exp(
                 -loss_coefficient * optical_path_length))
    return output


