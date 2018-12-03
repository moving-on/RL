"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain import NoisyNetDQN
import matplotlib.pyplot as plt
import pickle

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = NoisyNetDQN(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, replace_target_iter=200,
                  memory_size=2000, is_noisy=False)

total_steps = 0
sigma_his = []
loss_his = []
reward_his = []

for i_episode in range(200):

    observation = env.reset()
    ep_r = 0
    ep_step = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.perceive(observation, action, reward, observation_, done)

        ep_r += reward
        if total_steps > 100:
            #sigma, loss = RL.learn()
            #sigma_his.append(sigma)
            loss = RL.learn()
            loss_his.append(loss)

        ep_step += 1

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2),
                  'ep_step: ', ep_step)
            reward_his.append(ep_r)
            break

        observation = observation_
        total_steps += 1

with open('log', 'wb') as file:
    pickle.dump((sigma_his,loss_his,reward_his), file)

