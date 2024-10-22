from stable_baselines3.common.env_checker import check_env

from fabry_perot_server_momentan import Environment

env = Environment(history_length=4, noise_level=0.0002)

check_env(env)

