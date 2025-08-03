import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Fabry_perot.experimental_quantities import *
from Fabry_perot.calculate_output_powers import power_fabry_plotting

def PPO_clipped(x, epsilon):

    func = np.piecewise(
        x,
        [x <= 1 - epsilon, (x > 1 - epsilon) & (x < 1 + epsilon), x >= 1 + epsilon],
        [lambda x: 1 - epsilon, lambda x: x, lambda x: 1 + epsilon],
    )

    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(16, 9))
    plt.plot(x, func, label="PPO clipped function")
    plt.title("Clip Operator")

    # Adding horizontal lines at 1-epsilon and 1+epsilon
    plt.axvline(1 - epsilon, color="red", linestyle="--", label=f"1 - $\epsilon$")
    plt.axvline(1 + epsilon, color="green", linestyle="--", label=f"1 + $\epsilon$")

    # Setting labels for axes
    plt.xlabel("r")
    plt.ylabel("Function Value")

    # Adding custom tick labels for the y-axis
    plt.yticks(
        [1 - epsilon, 1, 1 + epsilon], [f"1 - $\epsilon$", "1", f"1 + $\epsilon$"]
    )
    plt.xticks(
        [1 - epsilon, 1, 1 + epsilon], [f"1 - $\epsilon$", "1", f"1 + $\epsilon$"]
    )

    # Showing the legend
    plt.legend()

    # plt.show()
    plt.savefig(
        "/home/robin/Dokumente/Masterarbeit/RL_for_cavity_control/result_images/PPO_clipped.png"
    )


plt.rcParams.update({"font.size": 24})


def power_transmitted_fabry(spectral_range):
    # Define ΔL values symmetrically around 0
    delta_L_values = np.linspace(
        -spectral_range / 8 * wavelength, spectral_range / 1.5 * wavelength, 10000
    )

    # Compute power output for each ΔL
    outputs_plotting = np.array(
        [power_fabry_plotting(delta_L) for delta_L in delta_L_values]
    )

    # Normalize by central value
    outputs_plotting /= power_fabry_plotting(0)

    # Plot the results
    plt.figure(figsize=(16, 9))
    plt.plot(delta_L_values / wavelength, outputs_plotting, label="Fabry_Plotting")

    # Compute resonance positions (multiples of λ/2)
    max_order = int(
        np.max(delta_L_values / (wavelength / 2))
    )  # Highest resonance order in the range
    resonance_positions = (
        np.arange(spectral_range / 8 * wavelength, max_order + 1) * 0.5
    )  # Resonances at multiples of λ/2

    # Ensure unique labels with integer values
    plt.xticks(
        resonance_positions, labels=[str(int(n * 2)) for n in resonance_positions]
    )

    plt.xlabel(r"Frequency(Free spectral range)")
    plt.ylabel("Transmission")
    # plt.legend(loc='upper left')
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.show()
    plt.savefig(
        "/home/robin/Dokumente/Masterarbeit/RL_for_cavity_control/result_images/FSR.png"
    )


