# Reinforcement Learning for Cavity Control 

A master's thesis project exploring reinforcement learning (RL) for stabilizing optical cavities, specifically Fabry-Pérot and coupled configurations, without relying on traditional error signals like Pound-Drever-Hall (PDH). The goal is to demonstrate that RL agents can maintain resonance purely from indirect physical observables, such as transmitted power, under realistic noise and uncertainty.

Optical cavities play a vital role in high-precision physics experiments, such as gravitational wave detection, laser stabilization, and quantum-enhanced sensing. Their effectiveness depends critically on keeping the cavity “locked”, maintaining resonance despite environmental noise and internal drift. Traditional approaches, such as PDH locking, use engineered error signals derived from phase-modulated feedback. While effective, these methods require complex hardware, precise calibration, and rigid modeling assumptions.

This project takes a different approach: it asks whether model-free RL agents can learn to lock an optical cavity using only minimal feedback, even in the absence of an engineered error signal. Through custom-built simulation environments that replicate Fabry-Pérot and coupled cavity physics, the agents learn to interact with noisy, partially observable systems using algorithms like PPO, SAC, and TQC.


## Installation
 

1. Clone the repository and install the necessary libaries

```bash

git clone https://github.com/robsyr/RL_for_cavity_control.git

cd RL_for_cavity_control

pip install -r requirements.txt


```


## File Descriptions

The repository is organized into subdirectories based on the type of cavity system and the purpose of the code:

- `Fabry_perot/` – Environment and training scripts for a single Fabry-Pérot cavity
- `Coupled_cavity/` – Environment and tools for a coupled cavity system
- `Plotting/` – Visualization utilities for thesis 

---

### `/Fabry_perot/` – Fabry-Pérot Cavity

Contains all code related to the simulation and training of agents on a single Fabry-Pérot cavity.

- `fabry_perot.py`  
    Defines the custom Gymnasium environment. This includes the state and action space, physics-based cavity model, reward function, and noise injection.

- `learning.py`
    Runs full training loops across multiple configurations. Sweeps over history_length, noise_level, and `action_size`, and trains agents using algorithms like TQC, SAC, and PPO. Saves models, evaluation logs, and TensorBoard data for later analysis.

- `calculate_output_powers.py`  
    Calculates reflected and transmitted power from the cavity based on the input parameters.

- `experimental_quantities.py`  
    Stores physical constants and experimental parameters such as laser wavelength, cavity length, and mirror reflectivity.

- `test_action_size.py`  
    Plots training curves grouped by `algorithm` and `action_size` for a fixed `history_length`. Applies smoothing and visualizes mean reward with shaded standard deviation. Useful for comparing how different action sizes affect training performance within a consistent setup.

- `learning_curves.py`  
    Plots smoothed learning curves for different `action_size` values, grouped by `algorithm` and filtered by `history_length`. Uses binning and exponential smoothing to compare how action resolution affects learning behavior. Outputs publication-ready LaTeX-style plots with scientific notation.
- `powerhold.py`  
    Compares transmission stability between two trained RL agents (with different history lengths) and a no-agent baseline. Simulates the same environment across trials, averages power outputs over multiple episodes, and plots smoothed power transmission curves with standard deviation shading.

- `test_filtered_by_action_size.py`  
    Plots smoothed reward curves across varying `history_length` values for a fixed `action_size` and algorithm. 

- `test_filtered_by_history_length.py`
    Compares different `action_size` values at fixed `history_length` and algorithm.

---


### `/Coupled_cavity/` – Coupled Cavity System

Includes the environment and analysis tools for a more complex cavity system involving two end mirrors and a central membrane.

- `coupled_env.py`  
    Custom Gym environment for a coupled cavity with multi-dimensional state and action spaces.

- `learning_coupled.py`
    Equivalent to `learning.py` but for the coupled cavity environment. Trains agents under varying system parameters and logs performance for each setup. Supports TQC, SAC, and PPO algorithms.

- `calculate_output_powers.py`  
    Calculates optical power values in the coupled cavity setup.

- `experimental_quantities.py`  
    Stores physical constants and experimental parameters such as laser wavelength, cavity length, and mirror reflectivity.

- `learning_curves_coupled.py`      
    Compares training performance across a range of tm values for a fixed set of hyperparameters. Applies binning and exponential smoothing to visualize trends and variability in learning progress for each tm.

- `plot_coupled_learning_variable.py`  
    Loads training logs and plots learning curves averaged over different tm values, grouped by algorithm, history length, and action size. Applies smoothing and displays standard deviation bands to show variability across runs.

---

### `Plotting/` – Thesis Figures

Includes the code, for figures used int the thesis

- `intuitive_control.py`  
    Shows how an RL agent could be capable to maintain a cavity lock, while just relying on output power and previously taken actions. 

- `plots_for_thesis.py`  
    Genereate some Plots used in this thesis. 
 

## License

This project is part of academic research. Please cite appropriately if using this work.

## Contact

For questions regarding this research, please contact the author through the GitHub repository.
 