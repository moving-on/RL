import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPO(object):
    def __init__(self, action_dim, state_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sess = tf.Session()
        self.state = tf.placeholder(tf.float32, [None, state_dim], "state")

        #critic
        with tf.variable_scope("critic"):
            l1 = tf.layers.dense(self.state, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.r = tf.placeholder(tf.float32, [None, 1], "reward")
            self.advantage = self.r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(learning_rate=C_LR).minimize(self.closs)

        #actor
        self.pi, self.pi_params = self.build_anet('pi', trainable=True)
        self.oldpi, self.oldpi_params = self.build_anet("oldpi", trainable=False)

        self.tfa = tf.placeholder(tf.int32, [None,], "action")
        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)
        oldpi_prob = tf.gather_nd(params=self.oldpi, indices=a_indices)
        self.tfadv = tf.placeholder(tf.float32, [None, 1], "advantage")
        with tf.variable_scope("loss"):
            with tf.variable_scope("surrogate"):
                self.ratio = pi_prob / oldpi_prob
                surr = self.ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(self.oldpi, self.pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * self.kl))
            else:
                self.aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(self.ratio, 1-METHOD['epsilon'], 1+METHOD['epsilon'])))
        self.atrain_op = tf.train.AdamOptimizer(learning_rate=A_LR).minimize(self.aloss)

        self.update_oldpi_op = [oldp.assign(p) for p,oldp in zip(self.pi_params, self.oldpi_params)]

        self.sess.run(tf.global_variables_initializer())
       
    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, feed_dict={self.state:s, self.r:r})

        #update actor
        if METHOD['name'] == "kl_pen":
            for _ in range(A_UPDATE_STEPS):
                aprob, aoldprob = self.sess.run([self.pi, self.oldpi], feed_dict={self.state:s})
                _, loss, kl, ratio = self.sess.run([self.atrain_op, self.aloss, self.kl_mean, self.ratio],
                        feed_dict={self.state:s, self.tfa:a, self.tflam:METHOD["lam"]}) 
                if kl > 4*METHOD['kl_target']:
                    break
            if kl < METHOD['kl_target'] / 1.5:
                METHOD['lam'] /= 2
            elif ki > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
        else:
            for _ in range(A_UPDATE_STEPS):
                _, loss, ratio, pi, oldpi = self.sess.run([self.atrain_op, self.aloss, self.ratio, self.pi, self.oldpi], 
                        feed_dict={self.state:s, self.tfa:a.ravel(), self.tfadv:adv})
                #print pi
                #print oldpi
                #print ratio

        for _ in range(C_UPDATE_STEPS):
            self.sess.run(self.ctrain_op, feed_dict={self.state:s, self.r:r})

    def choose_action(self, s):
        act_prob = self.sess.run(self.pi, feed_dict={self.state:s[np.newaxis, :]})
        action = np.random.choice(range(self.action_dim), p=act_prob.ravel())
        return action
        
    def build_anet(self, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(self.state, 100, tf.nn.relu, trainable=trainable)
            act_prob = tf.layers.dense(l1, self.action_dim, trainable=trainable)
            act_prob = tf.nn.softmax(act_prob)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return act_prob, params

    def get_v(self, s):
        v = self.sess.run(self.v, feed_dict={self.state:s[np.newaxis, :]})
        return v[0,0]
