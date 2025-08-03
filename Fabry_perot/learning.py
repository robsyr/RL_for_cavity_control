from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from fabry_perot import Environment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import TQC
import time
import os

algorithm_dict = {
    "PPO": PPO,
    "SAC": SAC,
    "TQC": TQC,

    # Possible other algorithms
    "DDPG": DDPG,
    "TD3": TD3,
    "A2C": A2C,

}

# Hyperparameters
history_length = [1,2,3,4,5,6,7,8]
algorithms = ['TQC', 'SAC', 'PPO']
noise_level = [4e-12]  
action_size = [4e-11, 5e-11] 
TOTAL_TIMESTEPS = 2000000
TIMESTEPS_PER_SAVE = 500000  
EVAL_FREQUENCY = 20000  # Increase eval frequency for reduced storage use

# Training Loop
for i in range(3):
    for algo in algorithms:
        for length in history_length:
            for level in noise_level:
                for size in action_size:

                    # Set up directories for models and logs
                    models_dir = f"models/{algo}_length_{length}_level_{level}_action_size_{size}_try_{i}_{int(time.time())}/"
                    logdir = f"logs/{algo}_length_{length}_level_{level}_action_size_{size}_try_{i}_{int(time.time())}/"

                    os.makedirs(models_dir, exist_ok=True)
                    os.makedirs(logdir, exist_ok=True)

                    # Environment setup
                    env = Monitor(Environment(history_length=length, noise_level=level, action_size=size), logdir)
                    env.reset()
                    
                    # Eval callback with higher eval frequency and no best model saving
                    eval_callback = EvalCallback(
                        env, 
                        best_model_save_path=logdir, 
                        log_path=logdir, 
                        eval_freq=EVAL_FREQUENCY, 
                        n_eval_episodes=5,  # Increase to evaluate more episodes per evaluation
                        verbose=1
                    )

                    # Model initialization with TensorBoard logging (reduced frequency for logging)
                    model = algorithm_dict[algo](
                        'MlpPolicy', env, verbose=1, 
                        tensorboard_log=f'{logdir}_tensorboard',
                        # n_steps=2048,  # Reduce to fewer steps for logging if needed
                        # batch_size=64
                    )

                    # Training loop 
                    iters = 0
                    while iters * TIMESTEPS_PER_SAVE < TOTAL_TIMESTEPS:
                        model.learn(total_timesteps=TIMESTEPS_PER_SAVE, callback=eval_callback, reset_num_timesteps=False)
                        model.save(f"{models_dir}/{TIMESTEPS_PER_SAVE * (iters + 1)}")
                        iters += 1

                    # Save final model after all training is done
                    model.save(f"{models_dir}/final_model")
