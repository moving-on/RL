import tensorflow as tf
import numpy as np

class PolicyGradient(object):
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
      
        self.sess = tf.Session()
        self.build_net()

        self.sess.run(tf.global_variables_initializer()) 
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def build_net(self):
        self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features])
        self.tf_act = tf.placeholder(tf.int32, [None,])
        self.tf_vt = tf.placeholder(tf.float32, None)
        l1 = tf.layers.dense(self.tf_obs, 10, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                             bias_initializer=tf.constant_initializer(0.1))
        all_act = tf.layers.dense(l1, self.n_actions, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                             bias_initializer=tf.constant_initializer(0.1))

        self.all_act_prob = tf.nn.softmax(all_act)
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_act) 
        self.loss = tf.reduce_mean(neg_log_prob) * self.tf_vt
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        gradient_all = optimizer.compute_gradients(self.loss)
        grads_vars = [v for (g, v) in gradient_all if g is not None]
        self.gradient = optimizer.compute_gradients(self.loss, grads_vars)
        self.grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in self.gradient]
        self.train_op = optimizer.apply_gradients(self.grads_holder)

    def choose_action(self, observation):
        act_prob = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs:observation[np.newaxis,:]})
        action = np.random.choice(range(act_prob.shape[1]), p=act_prob.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self.discount_and_norm_rewards()
        #print self.ep_obs
        #print self.ep_as
        #print self.ep_rs

        ep_grads = []
        for i in range(len(self.ep_obs)):
            ep_obs = self.ep_obs[i]
            ep_as = self.ep_as[i]
            gradient = self.sess.run(self.gradient, feed_dict={self.tf_obs:np.vstack(ep_obs), self.tf_act:np.array(ep_as), self.tf_vt:discounted_ep_rs_norm[i]})
            ep_grads.append(gradient)
        grad_sum = {}
        for i in range(len(self.grads_holder)):
            k = self.grads_holder[i][0]
            grad_sum[k] = sum([g[i][0] for g in ep_grads])
        self.sess.run(self.train_op, feed_dict=grad_sum)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros(len(self.ep_rs))
        for i in range(len(self.ep_rs)):
            r = 0
            for j in reversed(range(0, len(self.ep_rs[i]))):
                r += self.gamma * self.ep_rs[i][j]
            discounted_ep_rs[i] = r

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
