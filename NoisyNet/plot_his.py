import pickle
import matplotlib.pyplot as plt

with open('log', 'rb') as file:
    sigma_his, loss_his, reward_his = pickle.load(file)

plt.figure()
plt.plot(range(len(sigma_his)), sigma_his)
plt.show()

plt.figure()
plt.plot(range(len(reward_his)), reward_his)
plt.show()

plt.figure()
plt.plot(range(len(loss_his)), loss_his)
plt.show()
