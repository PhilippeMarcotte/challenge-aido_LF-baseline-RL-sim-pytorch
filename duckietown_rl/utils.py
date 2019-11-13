import random

import gym
import numpy as np
import torch


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done, dist, angle, next_dist, next_angle):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done, dist, angle, next_dist, next_angle))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done, dist, angle, next_dist, next_angle))


    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones , dists, angles, next_dists, next_angles= [], [], [], [], [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done, dist, angle, next_dist, next_angle = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))
            dists.append(np.array(dist, copy=False))
            angles.append(np.array(angle, copy=False))
            next_dists.append(np.array(next_dist, copy=False))
            next_angles.append(np.array(next_angle, copy=False))
            

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1),
            "dist": np.stack(dists).reshape(-1,1),
            "angle": np.stack(angles).reshape(-1,1),
            "next_dist": np.stack(next_dists).reshape(-1,1),
            "next_angle": np.stack(next_angles).reshape(-1,1)            
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            dist = lane_pose.dist        # Distance to lane center. Left is negative, right is positive.
            angle = lane_pose.angle_rad  # Angle from straight, in radians. Left is negative, right is positive.
            action = policy.predict(np.array(obs), dist, angle)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward
