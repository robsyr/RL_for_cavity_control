from stable_baselines3.common.env_checker import check_env
import numpy as np
from fabry_perot import *
import os
from stable_baselines3 import SAC, DDPG
import pandas as pd

def main():

    max_episode_steps = 500
    average_over = 10
    number_tries = 100
    timestamp = 1701075539
    num = 2420000




    save_name = f"testing\DDPG.csv"

    env = Environment()

    models_dir = f"models/1710424088/DDPG_0"
    logdir = f"logs/1710424088"
    model_path = f"models/1710424088/30000.zip"
    model = DDPG.load(model_path, tensorboard_log=logdir)
    model.set_env(env)

    data = {"step" : [i for i in range(max_episode_steps+1)]}
    df = pd.DataFrame(data)
    for j in range(number_tries):
        # print((j + 1) / number_tries)
        done = False
        obs, info = env.reset()
        power = obs[-1]
        powerj =[power]
        print(powerj)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            power = obs[-1]
            powerj.append(power)
            print(powerj)
        while len(powerj) < max_episode_steps+1:
            powerj.append(power)
            print(powerj)
        #print(powerj)
        df["power_"] = powerj
        df.to_csv(save_name)

if __name__ == '__main__':
    main()