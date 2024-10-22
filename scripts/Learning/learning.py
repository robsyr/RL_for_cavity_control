from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from fabry_perot import Environment
import time
import os

algorithm_dict = {
    "PPO": PPO,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3,
    "A2C": A2C,
}

history_length = [1, 2, 3, 4, 5, 6, 7, 8]
algorithms = ['PPO', 'SAC', 'DDPG', 'TD3', 'A2C']
noise_level = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001] # in terms of the wavelength



for level in noise_level:
    for algo in algorithms:
        for length in history_length:
            
            models_dir = f"models/{algo}_length_{length}_level_{level}_{int(time.time())}/"
            logdir = f"logs/{algo}_length_{length}_level_{level}_{int(time.time())}/"

            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            if not os.path.exists(logdir):
                os.makedirs(logdir)

            env = Environment(history_length=length, noise_level=level)
            env.reset()

            # model= PPO('MlpPoliucy', env, verbose=1, tensorboard_log=logdir)
            model = algorithm_dict[algo]('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

            TOTAL_TIMESTEPS = 1000000
            TIMESTEPS_PER_ITERATION = 10000
            iters = 0

            while iters * TIMESTEPS_PER_ITERATION < TOTAL_TIMESTEPS:
                iters += 1
                model.learn(total_timesteps=TIMESTEPS_PER_ITERATION, reset_num_timesteps=False, tb_log_name=f"PPO")
                model.save(f"{models_dir}/{TIMESTEPS_PER_ITERATION * iters}")
