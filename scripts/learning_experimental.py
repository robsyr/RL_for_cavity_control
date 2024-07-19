from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from fabry_perot_old_1 import Environment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import time
import os
from sb3_contrib import TQC

algorithm_dict = {
    "PPO": PPO,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3,
    "A2C": A2C,
}

history_length = [6, 7, 8]
algorithms = ['PPO', 'SAC', 'DDPG', 'TD3', 'A2C']
action_size = [0.005, 0.004, 0.003, 0.002, 0.001]

level='experimental'
for i in range(5):
    for length in history_length:
        for size in action_size:

            models_dir = f"experimental_noise//models//action_size_{size}//Length_{length}//PPO_length_{length}_try_{i}"
            logdir = f"experimental_noise//logs//action_size_{size}//Length_{length}//PPO_length_{length}_try_{i}"

            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            if not os.path.exists(logdir):
                os.makedirs(logdir)

            env = Monitor(Environment(history_length=length, noise_level=level, action_size=size), logdir)
            env.reset()
            eval_callback = EvalCallback(env, best_model_save_path=logdir, log_path=logdir, eval_freq=1000)

            # model= PPO('MlpPoliucy', env, verbose=1, tensorboard_log=logdir)
            model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f'{logdir}_tensorboard')

            TOTAL_TIMESTEPS = 1500000
            TIMESTEPS_PER_ITERATION = 10000
            iters = 0

            while iters * TIMESTEPS_PER_ITERATION < TOTAL_TIMESTEPS:
                iters += 1
                model.learn(total_timesteps=TIMESTEPS_PER_ITERATION, callback=eval_callback,
                            reset_num_timesteps=False,
                            tb_log_name=f"PPO")
                model.save(f"{models_dir}/{TIMESTEPS_PER_ITERATION * iters}")
