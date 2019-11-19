import random

import numpy as np
import torch
import gym
from gym import wrappers
import gym_duckietown
import os

from args import get_ddpg_args_train
from ddpg import DDPG
from utils import seed, evaluate_policy, ReplayBuffer
from wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, SteeringToWheelVelWrapper
from env import launch_env
import logging


policy_name = "DDPG"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = get_ddpg_args_train()

file_name = "{}_{}".format(
    policy_name,
    str(args.seed),
)

if not os.path.exists("./results"):
    os.makedirs("./results")
if args.save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

env = launch_env()

logging.getLogger('gym-duckietown').setLevel(logging.ERROR)


# Wrappers
env = ResizeWrapper(env)
env = NormalizeWrapper(env)
env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
env = ActionWrapper(env)
env = DtRewardWrapper(env)
env = SteeringToWheelVelWrapper(env)
env = wrappers.Monitor(env, './gym_results', video_callable=lambda episode_id: True, force=True)



# Set seeds
seed(args.seed)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


# Initialize policy

policy = DDPG(state_dim, action_dim, max_action)

replay_buffer = ReplayBuffer(args.replay_buffer_max_size)

# Evaluate untrained policy
# evaluations= [evaluate_policy(env, policy, device)]


total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
episode_reward = None
env_counter = 0
while total_timesteps < args.max_timesteps:
    if done:

        if total_timesteps != 0:
            print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (
                total_timesteps, episode_num, episode_timesteps, episode_reward))
            policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(env, policy, device))

            if args.save_models:
                policy.save(file_name, directory="./pytorch_models")
            np.savez("./results/{}.npz".format(file_name),evaluations)

        # Reset environment
        env_counter += 1
        # env.close()
        obs = env.reset()
        # env.reset_video_recorder()
        # env = wrappers.Monitor(env, './gym_results', video_callable=lambda episode_id: True, force=True)
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    dist = lane_pose.dist        # Distance to lane center. Left is negative, right is positive.
    angle = lane_pose.angle_rad  # Angle from straight, in radians. Left is negative, right is positive.
    # Select action randomly or according to policy
    if total_timesteps < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.predict(np.array(obs), np.array(dist), np.array(angle),
         only_pid=total_timesteps-args.start_timesteps<args.pid_timesteps)
        if args.expl_noise != 0:
            action = (action + np.random.normal(
                0,
                args.expl_noise,
                size=env.action_space.shape[0])
                      ).clip(env.action_space.low, env.action_space.high)

    # Perform action
    new_obs, reward, done, _ = env.step(action)
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    next_dist = lane_pose.dist       # Distance to lane center. Left is negative, right is positive.
    next_angle = lane_pose.angle_rad  # Angle from straight, in radians. Left is negative, right is positive.

    if episode_timesteps >= args.env_timesteps:
        done = True

    done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
    episode_reward += reward

    # Store data in replay buffer
    replay_buffer.add(obs, new_obs, action, reward, done_bool, dist, angle, next_dist, next_angle)

    obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(env, policy, device))


if args.save_models:
    policy.save(file_name, directory="./pytorch_models")
np.savez("./pytorch_models/{}.npz".format(file_name),evaluations)
