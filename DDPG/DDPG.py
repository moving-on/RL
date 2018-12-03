import tensorflow as tf
import numpy as np
import gym
import time

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('actor'):
            self.a = self.build_net(S, scope="eval_net", trainable=True)
            self.a_ = self.build_net(S_, scope="target_net", trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor/eval_net")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor/target_net")

        if self.replacement["name"] == "hard":
            self.t_replace_counter = 0
            self.replace_hard = [tf.assign(t, e) for (t,e) in zip(self.t_params, self.e_params)]
        elif self.replacement["name"] == "soft":
            self.replace_soft = [tf.assign(t, self.replacement["tau"] * e + (1-self.replacement["tau"]) * t) for (t,e) in zip(self.t_params, self.e_params)]

    def build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            l1 = tf.layers.dense(inputs=s, units=30, activation=tf.nn.relu,
                                 kernel_initializer=init_w,
                                 bias_initializer=init_b,
                                 trainable=trainable)
            actions = tf.layers.dense(l1, self.a_dim, activation=tf.nn.tanh,
                                      kernel_initializer=init_w,
                                      bias_initializer=init_b,
                                      trainable=trainable)
            scaled_a = tf.multiply(actions, self.action_bound)
        return scaled_a

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={S:s})[0] 

    def add_grad_to_graph(self, a_grads):
        self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
        optimizer = tf.train.AdamOptimizer(-self.lr)
        self.train_op = optimizer.apply_gradients(zip(self.policy_grads, self.e_params))

    def learn(self, s):
        self.sess.run(self.train_op, feed_dict={S:s})
    
        if self.replacement["name"] == "soft":
            self.sess.run(self.replace_soft)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.replace_hard)
            self.t_replace_counter += 1

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.a_dim = action_dim
        self.s_dim = state_dim
        self.gamma = gamma
        self.lr = learning_rate
        self.replacement = replacement

        self.a = a
        with tf.variable_scope('critic'):
            self.q = self.build_net(S, self.a, scope="eval_net", trainable=True)
            self.q_ = self.build_net(S_, a_, scope="target_net", trainable=False)

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic/eval_net")
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic/target_net")

        self.target_q = R + self.gamma * self.q_
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.a_grads = tf.gradients(self.q, self.a)[0]

        if self.replacement["name"] == "hard":
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for (t,e) in zip(self.t_params, e_params)]
        else:
            self.soft_replacement = [tf.assign(t, self.replacement["tau"]*e+(1-self.replacement["tau"])*t) for (t,e) in zip(self.t_params, self.e_params)]

    def build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., .1)
            init_b = tf.constant_initializer(0.1)

            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            q = tf.layers.dense(l1, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S:s, S_:s_, R:r, self.a:a})
        if self.replacement["name"] == "hard":
            if self.t_replace_counter % self.replacement["rep_iter_c"] == 0:
                self.sess.run(self.hard_replacement)
                self.t_replace_counter += 1
        else:
            self.sess.run(self.soft_replacement)

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

var = 3
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()
        
        a = actor.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        M.store_transition(s, a, r / 10, s_)

        if M.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break

print('Running time: ', time.time()-t1)

