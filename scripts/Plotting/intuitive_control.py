import numpy as np
import matplotlib.pyplot as plt
from scripts.Fabry_perot.calculate_output_powers import power_fabry_plotting
from scripts.Fabry_perot.experimental_quantities import wavelength

# # Generate x values for Fabry-Perot function
x = np.linspace(-0.02 * wavelength, 0.02 * wavelength, 10000)
y = power_fabry_plotting(x)

# Define key points
x_red_start = 0  # Red arrow starts at x = 0 (peak)
y_red_start = max(y)  # Maximum power value
y_red_end = 0.9 * y_red_start  # Red arrow ends at 90% of max power

# Find the x value corresponding to y_red_end
x_red_end_index = np.argmin(np.abs(y - y_red_end))
x_red_end = x[x_red_end_index]  # x position where power drops to 90%

y_blue_end = 0.8 * y_red_start  # Blue arrows end at 80% of max power
x_blue_end_index = np.argmin(np.abs(y - y_blue_end))
x_blue_end = x[x_blue_end_index]  # x position for 80% power

# Background Gaussian (rescaled properly)
x_gauss = np.linspace(-2, 2, 400)  # Standard Gaussian range (before scaling)
gauss_width = (x_blue_end - x_red_end) * 3.5  # Adjust spread to match function width
y_gauss = np.exp(-x_gauss**2) * (0.15 * y_red_start) + y_red_end  # Scale and shift vertically
x_gauss_shifted = x_gauss * gauss_width + x_red_end  # Scale and shift horizontally

plt.rcParams.update({'font.size': 18})

# Create plot
fig, ax = plt.subplots(figsize=(16, 9))

# Background Gaussian (now scaled correctly)
gauss_patch = ax.fill_between(x_gauss_shifted, y_gauss, y_red_end, alpha=0.1, color="blue", label="Probability distribution")

# Fabry-Perot Power Curve
ax.plot(x, y, 'k-', label="Fabry-Perot Power")

# Draw red arrow (from max y to 90% max y)
red_arrow = ax.annotate("", xy=(x_red_end, y_red_end), xytext=(x_red_start, y_red_start),
                         arrowprops=dict(arrowstyle="->", color="red", linewidth=2, label="Controlled displacement"))

# Draw blue arrows (symmetric movement down)
blue_arrow1 = ax.annotate("", xy=(-x_blue_end, y_blue_end), xytext=(x_red_end, y_red_end),
                           arrowprops=dict(arrowstyle="->", color="blue", linewidth=2))
blue_arrow2 = ax.annotate("", xy=(x_blue_end, y_blue_end), xytext=(x_red_end, y_red_end),
                           arrowprops=dict(arrowstyle="->", color="blue", linewidth=2))

# Draw horizontal line at blue arrow ends (extending beyond the graph limits)
extra_space = 1.5 * x_blue_end  # Extend further than x limits
ax.hlines(y_blue_end, -extra_space, extra_space, colors='black', linestyles='dashed')
ax.scatter(0, 1, color='black', s=100, zorder=3, label="Initial Displacement")  # Black dot

# Annotate percentages
ax.text(-x_blue_end, y_blue_end - 0.02 * y_red_start, "13%", color='blue', fontsize=16, ha='center')
ax.text(x_blue_end, y_blue_end - 0.02 * y_red_start, "87%", color='blue', fontsize=16, ha='center')

# Labels and Formatting
ax.set_xlabel(r"Detuning $\delta = \chi - u$")
ax.set_ylabel("Power")
ax.axhline(0, color='black', linewidth=1)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_ylim(0.7, 1.2)
ax.set_xlim(-1e-8, 1e-8)

# Create legend
legend_elements = [
    plt.Line2D([0], [0], color='black', marker='o', markersize=10, linestyle='', label="Initial displacement"),
    plt.Line2D([0], [0], color='red', lw=2, label="Controlled displacement"),
    plt.Line2D([0], [0], color='blue', lw=2, label="New Candidates"),
    plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.2, label="Probability Distribution")
]
ax.legend(handles=legend_elements, loc="upper left")
ax.set_xticks([])  # Remove tick marks
ax.set_xticklabels([])  # Remove tick labels

# Save and show
plt.savefig('/home/robin/Dokumente/Masterarbeit/MA/Images/Control_intuitive.png', bbox_inches='tight')
plt.show()






# Create the figure
plt.rcParams.update({'font.size': 18})

fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(x, y, 'k', label=None)

# Define the arrow starting and ending points
y_start = max(y)  # Start at max power
y_end = 0.9 * y_start  # End at 90% power

# Find x corresponding to y_end
xend_idx = np.argmin(np.abs(y - y_end))
xend = x[xend_idx]

# Background Gaussian (rescaled properly)
x_gauss = np.linspace(-2, 2, 400)  # Standard Gaussian range (before scaling)
gauss_width = (xend) * 2.5  # Adjust spread to match function width
y_gauss = np.exp(-x_gauss**2) * (0.15 * y_start) + 1  # Shift the distribution above 1
x_gauss_shifted = x_gauss * gauss_width   # Scale and shift horizontally

# Custom legend entries
legend_elements = [
    plt.Line2D([0], [0], color='black', marker='o', markersize=10, linestyle='', label="Initial Displacement"),
    plt.Line2D([0], [0], color='blue', lw=2, label="New Candidates"),
    plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.1, label="Probability Distribution")
]
# Background Gaussian (now scaled correctly)
gauss_patch = ax.fill_between(x_gauss_shifted, y_gauss, y_start, alpha=0.1, color="blue", label="Probability distribution")

ax.scatter(0, 1, color='black', s=100, zorder=3, label="Initial Displacement")  # Black dot
# Draw arrows
ax.annotate('', xy=(xend, y_end), xytext=(0, y_start),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('', xy=(-xend, y_end), xytext=(0, y_start),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# Draw text annotations at proper positions
ax.text(xend, y_end - 0.02 * y_start, '50%', color='blue', fontsize=16, ha='center')
ax.text(-xend, y_end - 0.02 * y_start, '50%', color='blue', fontsize=16, ha='center')
ax.hlines(y_end, -extra_space, extra_space, colors='black', linestyles='dashed')


# Labels and formatting
ax.set_xlabel(r'Detuning $\delta = \chi - u$')
ax.set_ylabel('Power')
ax.set_ylim(0.7, 1.2)
ax.set_xlim(-1e-8, 1e-8)
ax.legend(handles=legend_elements, loc="upper left")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xticks([])  # Remove tick marks
ax.set_xticklabels([])  # Remove tick labels


plt.savefig('/home/robin/Dokumente/Masterarbeit/MA/Images/Control_intuitive_symm.png', bbox_inches='tight')

# Show the plot
plt.show()
