import matplotlib.pyplot as plt
import numpy as np
import time
from experimental_quantities import *




    








def power_output_transmitted_coupled_cavity(action1, action2, noise1, noise2, tm):
    rin = 0.99
    tin=1-rin
    rout=0.99
    tout=1-rout
    rm=1-tm
    k = 2*np.pi/wavelength
    L = length_interferometer
    xm = 0.5*length_interferometer

    denominator = (1  + (rm+tm)*(
        np.sqrt(rin*rout)*2*np.cos(2*k*(L+noise1+noise2+action1+action2)) -np.sqrt(rm*rout)*rin*2*np.cos(2*k*(xm+noise2+action2)) -
        np.sqrt(rin*rm)*rout*2*np.cos(2*k*(L+noise1+action1-xm)) +rin*rout*(rm+tm)
        )
        - np.sqrt(rm*rin)*2*np.cos(2*k*(L+noise1+action1-xm)) - np.sqrt(rm*rout)*2*np.cos(2*k*(xm+noise2+action2)) 
        + rm*np.sqrt(rin*rout)*2*np.cos(2*k*(L+noise1+action1-2*xm-noise2-action2)) +rm*(rin+rout)
        )
    return tin*tout*tm / denominator


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





