#!/usr/bin/env python3
"""Simulator-side collection interface smoke test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'simulator'))

from dexjoco.benchmark6.dataset import pack_action, public_observation
from dexjoco.benchmark6.environments import make_env


def main():
    env = make_env("hammer_nail", "allegro", seed=0, pixels=False)
    try:
        observation = public_observation(env)
        action = pack_action(env, env.hold_action())
        env.step(env.hold_action())
        assert observation["state"].shape == (2, 31)
        assert action["action"].shape == (2, 28)
        print("collection interface smoke: passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
