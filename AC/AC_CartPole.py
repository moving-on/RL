import tensorflow as tf
import numpy as np
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr

        self.s = tf.placeholder(tf.float32, [1, n_features])
        self.a = tf.placeholder(tf.int32, None)
        self.td_error = tf.placeholder(tf.float32, None)

        with tf.variable_scope("actor"):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., 1.),
                bias_initializer=tf.constant_initializer(0.1)
             )

            self.act_prob = tf.layers.dense(
                inputs=l1,
                units=self.n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1)
            )

        log_prob = tf.log(self.act_prob[0, self.a])
        self.exp_v = tf.reduce_mean(log_prob * self.td_error)
   
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict={self.s:s[np.newaxis, :], self.a:a, self.td_error:td})
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.act_prob, feed_dict={self.s:s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self, sess, n_features, n_actions, lr=0.01):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr 
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, self.n_features])
        self.v_ = tf.placeholder(tf.float32, [1, 1])
        self.r = tf.placeholder(tf.float32, None)

        with tf.variable_scope("critic"):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1)
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1)
            )

        self.td_error = self.r + GAMMA * self.v_ - self.v
        self.loss = tf.square(self.td_error)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s:s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], feed_dict={self.s:s, self.v_:v_, self.r:r})
        return td_error

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, n_actions=N_A, lr=LR_C)

sess.run(tf.global_variables_initializer())

for i in range(MAX_EPISODE):
    s= env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER:
            env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done:
            r -= 20
        track_r.append(r)

        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1

        if done or t>= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i, "  reward:", int(running_reward))
            break
