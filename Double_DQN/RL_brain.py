import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Memory(object):
    def __init__(self, memory_size, n_features):
        self.memory_size = memory_size
        self.n_features = n_features
        self.buffer = np.zeros((self.memory_size, n_features*2+2))

    def add(self, sample, index):
        self.buffer[index,:] = sample

    def get(self, sample_index):
        return self.buffer[sample_index, :]

class DeepQNetwork(object):
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=300,
                 memory_size=500, batch_size=32, e_greedy_increment=None, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(memory_size, n_features)

        self.build_net()
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
        t_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")

        self.update_target = [tf.assign(t, e) for (t,e) in zip(t_params, e_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print e_params
        print t_params

        self.cost_his = []

    def build_net(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features])
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions])
        with tf.variable_scope("eval_net"):
            l1 = tf.layers.dense(self.state, 10, tf.nn.relu)
            self.q_eval = tf.layers.dense(l1, self.n_actions)

        with tf.variable_scope("target_net"):
            l2 = tf.layers.dense(self.state, 10, tf.nn.relu)
            self.q_next = tf.layers.dense(l2, self.n_actions)
        self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_eval)
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            q_value = self.sess.run(self.q_eval, feed_dict={self.state:state})
            action = np.argmax(q_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
        
    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_size
        transition = np.hstack((s, [a,r], s_))
        self.memory.add(transition, index)
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            #print self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")[0])
            #print self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")[0])
            self.sess.run(self.update_target)
            print "target_params update"
            #print self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")[0])
            #print self.sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")[0])

        if self.memory_counter < self.memory_size:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch = self.memory.get(sample_index)

        q_next = self.sess.run(self.q_next, feed_dict={self.state:batch[:, -self.n_features:]})
        q_next_eval = self.sess.run(self.q_eval, feed_dict={self.state:batch[:, -self.n_features:]})
        q_eval = self.sess.run(self.q_eval, feed_dict={self.state:batch[:, :self.n_features]})

        q_target = q_eval.copy()

        reward = batch[:, self.n_features+1]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        act_index = batch[:, self.n_features].astype(int)
        max_act_next = np.argmax(q_next_eval, axis=1)
        q_act_next = q_next[batch_index, max_act_next]
        q_target[batch_index, act_index] = reward + self.gamma * q_act_next

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.state:batch[:, :self.n_features], self.q_target:q_target})

        self.learn_step_counter += 1
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
        self.cost_his.append(cost)

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show() 
