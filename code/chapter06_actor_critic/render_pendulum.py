"""
6： A2C Pendulum-v1 

：
    python render_pendulum.py --model output/actor_critic_pendulum.zip
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    parser = argparse.ArgumentParser(description=" Pendulum A2C ")
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--vecnormalize", type=str,
                        default="output/actor_critic_pendulum_vecnormalize.pkl",
                        help="VecNormalize ")
    parser.add_argument("--output", type=str, default="output/pendulum_actor_critic.gif",
                        help="GIF ")
    parser.add_argument("--seed", type=int, default=0, help=" seed")
    parser.add_argument("--max-steps", type=int, default=200, help="")
    parser.add_argument("--fps", type=int, default=30, help="GIF ")
    args = parser.parse_args()

    model = A2C.load(args.model)
    render_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    vec_env = DummyVecEnv([lambda: render_env])
    vec_env = VecNormalize.load(args.vecnormalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.seed(args.seed)
    obs = vec_env.reset()
    frames = []
    total_reward = 0.0

    for _ in range(args.max_steps):
        frames.append(render_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += float(reward[0])
        if done[0]:
            break

    vec_env.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1000 / args.fps, loop=0)
    print(f": {total_reward:.1f}")
    print(f"GIF  {output_path}")


if __name__ == "__main__":
    main()
