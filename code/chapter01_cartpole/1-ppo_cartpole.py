"""
1： Stable-Baselines3  PPO  CartPole

 SwanLab （、），
 GUI 。

：
    # ： + SwanLab （ GUI，）
    python 1-ppo_cartpole.py

    #  GUI （）
    python 1-ppo_cartpole.py --gui

 --gui ：
     headless（）， GUI 。
    --gui  CartPole 。
     GUI ，（~16ms），；
     GUI ，，。
"""

import argparse
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import HumanOutputFormat
from swanlab.integration.sb3 import SwanLabCallback
import swanlab


class LogApproxKL(BaseCallback):
    """ train/approx_kl  SwanLab。

    SB3  PPO.train()  logger.record("train/approx_kl", ...) ，
     numpy.float32 。SwanLab  SB3  write() 
    isinstance(value, (int, float)) ， numpy.float32 
    （numpy.float64  Python float ）， approx_kl 。

     train() ， logger  approx_kl ，
     Python float  swanlab.log 。
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # train()  _on_rollout_end ，
        # logger  train 。
        logger = self.model.logger
        if hasattr(logger, "name_to_value") and "train/approx_kl" in logger.name_to_value:
            value = float(logger.name_to_value["train/approx_kl"])
            swanlab.log({"train/approx_kl": value}, step=self.num_timesteps)


class RestoreStdoutLog(BaseCallback):
    """ SB3 。

    SwanLabCallback._init_callback()  self.model.set_logger(...)，
    " SwanLab" logger  SB3  logger，
     stdout  ep_rew_mean / fps / approx_kl 
    HumanOutputFormat（ verbose=1 ）。

     _init_callback （ SwanLabCallback  logger），
     logger  stdout ，，
     SwanLab 。 callback  SwanLabCallback 。
    """

    def _init_callback(self) -> None:
        # SwanLabCallback  logger  SwanLabOutputFormat，
        #  stdout ， verbose=1 。
        self.model.logger.output_formats.append(HumanOutputFormat(sys.stdout))

    def _on_step(self) -> bool:
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 PPO CartPole ")
    parser.add_argument(
        "--gui", action="store_true",
        help=" GUI （，）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    # ==========================================
    # ：
    # ==========================================
    env = gym.make("CartPole-v1")

    # （、、）
    print("=" * 50)
    print("CartPole-v1 ")
    print("=" * 50)
    print(f"  :  {env.observation_space}")
    print(f"  :  {env.action_space}")
    print(f"  :  {env.observation_space.high}")
    print(f"  :  {env.observation_space.low}")
    print(f"  :   > ±{env.unwrapped.x_threshold}, "
          f" > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
          f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)")
    print("=" * 50)

    model = PPO("MlpPolicy", env, verbose=1)

    print("（ SwanLab ）...")
    swanlab_cb = SwanLabCallback(
        project="cartpole-ppo",
        experiment_name="PPO-CartPole-v1",
        mode="local",
    )
    model.learn(
        total_timesteps=80000,
        callback=[swanlab_cb, RestoreStdoutLog(), LogApproxKL()],
    )

    # 
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"！: {mean_reward} +/- {std_reward}")

    model.save("output/ppo_cartpole")
    env.close()

    # ==========================================
    # ：
    # ==========================================
    print("\n...")
    render_mode = "human" if args.gui else None
    vis_env = gym.make("CartPole-v1", render_mode=render_mode)
    model = PPO.load("output/ppo_cartpole")

    for episode in range(5):
        obs, info = vis_env.reset()
        done, truncated, score = False, False, 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = vis_env.step(action)
            score += reward
        print(f"   {episode + 1} : {score}")

    vis_env.close()

    if args.gui:
        print("\nGUI 。")
    else:
        print("\n:  --gui 。")

    print("SwanLab : swanlab watch swanlog")


if __name__ == "__main__":
    main()
