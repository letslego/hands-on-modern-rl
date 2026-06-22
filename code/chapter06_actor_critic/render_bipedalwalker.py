"""
6： A2C  BipedalWalker-v3 

：
    python render_bipedalwalker.py --model output/actor_critic_bipedalwalker.zip
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import A2C


def render_episode(model, max_steps=1600, seed=None):
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    episode_steps = 0
    for step in range(max_steps):
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episode_steps = step + 1
        if terminated or truncated:
            break
    env.close()
    return frames, total_reward, episode_steps


def downsample_frames(frames, max_frames):
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description=" A2C BipedalWalker ")
    parser.add_argument("--model", type=str, required=True, help=" A2C ")
    parser.add_argument("--output-dir", type=str, default="output/bipedalwalker_a2c_episodes",
                        help="GIF ")
    parser.add_argument("--episodes", type=int, default=3, help=" episode ")
    parser.add_argument("--fps", type=int, default=30, help="GIF ")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help=" episode  seed（）")
    parser.add_argument("--max-steps", type=int, default=1600,
                        help=" episode ")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="GIF （）")
    args = parser.parse_args()

    print(f": {args.model}")
    model = A2C.load(args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or [None] * args.episodes

    for ep, seed in enumerate(seeds[:args.episodes]):
        frames, reward, steps = render_episode(
            model, max_steps=args.max_steps, seed=seed
        )
        gif_frames = downsample_frames(frames, args.max_frames)
        seed_label = f", seed={seed}" if seed is not None else ""
        print(f"Episode {ep + 1}: reward={reward:.1f}, steps={steps}, "
              f"gif_frames={len(gif_frames)}{seed_label}")
        out_path = output_dir / f"bipedalwalker_a2c_ep{ep + 1}.gif"
        imageio.mimsave(out_path, gif_frames, duration=1000 / args.fps, loop=0)
        print(f"  Saved to {out_path}")

    print(f"\n GIF : {output_dir}")


if __name__ == "__main__":
    main()
