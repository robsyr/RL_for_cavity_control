import gymnasium as gym
from stable_baselines3 import PPO
import os
import time
import optuna
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

from fabry_perot_old_1 import *

# Define the hyperparameters to search over
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'n_steps': [128, 256, 512],
    'batch_size': [32, 64, 128],
    'ent_coef': [0.01, 0.1, 0.2],
    'gamma': [0.99, 0.98, 0.97],
}

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

def evaluate_policy(model, env, n_eval_episodes):
    episode_rewards = []
    power_outputs = []
    observations = []
    episode_reward = 0
    observation, info = env.reset()
    for step in range(n_eval_episodes):
        observations.append(observation)
        action, states = model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_rewards.append(episode_reward)
        power_outputs.append(observation[-1])
    mean_reward = np.mean(episode_rewards)
    return mean_reward, power_outputs

# Create the model

# hyperparameters to optimize
def optimize_ppo(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.)
    }


def optimize_agent(trial):
    """ Train the model and optimize
            Optuna maximises the negative log likelihood, so we
            need to negate the reward here
        """
    model_params = optimize_ppo(trial)
    env = Environment(history_length=5, action_size=0.0009, noise_level='experimental')
    model = PPO('MlpPolicy', env, verbose=0, **model_params)
    model.learn(600000)
    mean_reward, power_outputs = evaluate_policy(model, env, n_eval_episodes=10)

    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=4)
        best_params = study.best_params
        print(best_params)

        # Write results to CSV
        with open('his_5_optuna_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['param_name', 'param_value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for key, value in best_params.items():
                writer.writerow({'param_name': key, 'param_value': value})

        print("Results saved to optuna_results.csv")

    except KeyboardInterrupt:
        print('Interrupted by Keyboard')





"""
class PPOWrapper(BaseEstimator):
    def __init__(self, env):
        self.env = env
        self.model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    def fit(self, X, y=None):
        self.model.learn(total_timesteps=65000, callback=None)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y=None):
        pass


# Define the environment
env = Environment(history_length=3)
env.reset()

# Create PPO wrapper
ppo_wrapper = PPOWrapper(env)
# Perform the grid search
grid_search = GridSearchCV(ppo_wrapper, param_grid, cv=3, n_jobs=-1)
grid_search.fit(env.observation_space.sample(), env.action_space.sample())

# best model
best_model = grid_search.best_estimator_

# Train the best model
best_model.learn(total_timesteps=65000)

"""



