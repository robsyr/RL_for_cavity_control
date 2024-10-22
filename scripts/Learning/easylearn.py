from stable_baselines3 import PPO, DQN, SAC, DDPG, TD3, A2C
import os

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from fabry_perot import Environment
import time



models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = Monitor(Environment(history_length=3, noise_level=0.00005), logdir)
env.reset()
eval_callback = EvalCallback(env, best_model_save_path=logdir, log_path=logdir, eval_freq=1000)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TOTAL_TIMESTEPS = 1000000
TIMESTEPS_PER_ITERATION = 10000
iters = 0

while iters * TIMESTEPS_PER_ITERATION < TOTAL_TIMESTEPS:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS_PER_ITERATION,  callback=eval_callback, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS_PER_ITERATION * iters}")
