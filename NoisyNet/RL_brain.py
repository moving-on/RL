# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools
from tensorflow.python.framework import ops

def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper

class NoisyNetDQN:
    def __init__(self, n_actions, n_features, learning_rate, replace_target_iter, memory_size, is_noisy=True):
        self.sess = tf.InteractiveSession()
        self.is_noisy= is_noisy
        # init experience replay
        self.replay_buffer = deque(maxlen=memory_size)
        self.time_step = 0
        self.state_dim = n_features
        self.action_dim = n_actions
        self.learning_rate = learning_rate
        self.gamma = 0.9
        self.batch_size = 32
        self.replace_target_iter = replace_target_iter
        print 'state_dim:', self.state_dim
        print 'action_dim:', self.action_dim

        self.eval_input = tf.placeholder("float", [None, self.state_dim])
        self.select_input = tf.placeholder("float", [None, self.state_dim])

        self.y_input = tf.placeholder("float", [None, self.action_dim])

        self.epsilon_max = 0.9
        self.epsilon_increment = 0.001
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

        self.Q_select
        self.Q_eval

        self.loss
        self.optimize
        self.update_target_net

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        self.sigma_his = []

    def normal_dense(self, inputs, units, bias_shape, c_names, w_i, b_i=None, trainable=True, activation_fn=None):
        weights = tf.get_variable('weights', shape=[inputs.shape[1], units], initializer=w_i, trainable=trainable)
        bias = tf.get_variable('bias', shape=bias_shape, initializer=b_i, trainable=trainable)
        dense = tf.matmul(inputs, weights) + bias 
        if activation_fn is None:
            return dense
        else:
            return activation_fn(dense)

    def noisy_dense(self, x, input_size, output_size, name, trainable, activation_fn=tf.identity):
 
        # the function used in eq.7,8
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        # Initializer of \mu and \sigma
        mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(input_size, 0.5),
                                                    maxval=1*1/np.power(input_size, 0.5))
        sigma_init = tf.constant_initializer(0.4/np.power(input_size, 0.5))
        # Sample noise from gaussian
        p = tf.random_normal([input_size, 1])
        q = tf.random_normal([1, output_size])
        f_p = f(p); f_q = f(q)
        w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

        # w = w_mu + w_sigma*w_epsilon
        w_mu = tf.get_variable(name + "/w_mu", [input_size, output_size],
                initializer=mu_init, trainable=trainable)
        w_sigma = tf.get_variable(name + "/w_sigma", [input_size, output_size],
                initializer=sigma_init, trainable=trainable)
        w = w_mu + tf.multiply(w_sigma, w_epsilon)
        ret = tf.matmul(x, w)

        # b = b_mu + b_sigma*b_epsilon
        b_mu = tf.get_variable(name + "/b_mu", [output_size],
                initializer=mu_init, trainable=trainable)
        b_sigma = tf.get_variable(name + "/b_sigma", [output_size],
                initializer=sigma_init, trainable=trainable)
        b = b_mu + tf.multiply(b_sigma, b_epsilon)
        return activation_fn(ret + b)

    def build_layers(self, state, c_names, units, w_i, b_i, reg=None, trainable=False):
        if self.is_noisy:
            with tf.variable_scope('dense1'):
                dense1 = self.noisy_dense(state, name='dense1',
                    input_size=4, output_size=10, trainable=trainable, activation_fn=tf.nn.relu)
            with tf.variable_scope('dense2'):
                dense2 = self.noisy_dense(dense1, name='dense2',
                    input_size=10, output_size=self.action_dim, trainable=trainable)
        else:
            with tf.variable_scope('dense1'):
                dense1 = self.normal_dense(state, units, [units], c_names, w_i, b_i, trainable=trainable, activation_fn=tf.nn.relu)
            with tf.variable_scope('dense2'):
                dense2 = self.normal_dense(dense1, self.action_dim, [self.action_dim], c_names, w_i, b_i, trainable=trainable)
        return dense2

    @lazy_property
    def Q_select(self):
        with tf.variable_scope('select_net'):
            c_names = ['select_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.2)
            result = self.build_layers(self.select_input, c_names, 10, w_i, b_i, trainable=True)
            return result

    @lazy_property
    def Q_eval(self):
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_i = tf.random_uniform_initializer(-0.1, 0.1)
            b_i = tf.constant_initializer(0.1)
            result = self.build_layers(self.eval_input, c_names, 10, w_i, b_i, None, trainable=False)
            return result

    @lazy_property
    def loss(self):
        loss = tf.reduce_mean(tf.squared_difference(self.Q_select, self.y_input))
        return loss

    @lazy_property
    def optimize(self):
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss)

    @lazy_property
    def update_target_net(self):
        select_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'select_net')
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        return [tf.assign(e, s) for e, s in zip(eval_params, select_params)]

    def save_model(self):
        print("Model saved in : ", self.saver.save(self.sess, './checkpoint'))

    def restore_model(self):
        self.saver.restore(self.sess, './checkpoint')
        print("Model restored.")

    def perceive(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, update=True):
        """
        :param update: True means the action "update_target_net" executes outside, and can be ignored in the function
        """
        if update and self.time_step % self.replace_target_iter == 0:
            #print "select_net"
            #print self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="select_net")[0])
            #print "eval_net before"
            #print self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")[0])
            print "target replaced"
            self.sess.run(self.update_target_net)
            #print "eval_net after"
            #print self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")[0])
        #print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="select_net")
        #print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")

        self.time_step += 1
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        np.random.shuffle(minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: next_state_batch})
        Q_select = self.Q_select.eval(feed_dict={self.select_input: state_batch})

        # convert true to 1, false to 0
        done = np.array(done) + 0

        y_batch = Q_select.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        y_batch[batch_index, action_batch] = reward_batch + (1 - done) * self.gamma * np.max(Q_eval, axis=1)

        #print Q_select
        #print y_batch
        feed_dict = {
            self.y_input: y_batch,
            self.select_input: state_batch,
        }
        self.sess.run(self.optimize, feed_dict=feed_dict)
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        #sigma = self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="select_net")[1])[0][0]
        return loss

    def noisy_action(self, state):
        return np.argmax(self.Q_select.eval(feed_dict={self.select_input: [state]})[0])

    def normal_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            q_value = self.sess.run(self.Q_select, feed_dict={self.select_input: state})
            action = np.argmax(q_value)
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def choose_action(self, state):
        if self.is_noisy:
            return self.noisy_action(state)
        else:
            return self.normal_action(state)
