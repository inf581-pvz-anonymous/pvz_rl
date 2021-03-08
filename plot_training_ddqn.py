import numpy as np
import matplotlib.pyplot as plt
import sys
import torch


if __name__ == "__main__":
    name = sys.argv[1]
    rewards = np.load(name+"_rewards.npy")
    iterations = np.load(name+"_iterations.npy")
    loss = torch.load(name+"_loss")
    real_rewards = np.load(name+"_real_rewards.npy")
    real_iterations = np.load(name+"_real_iterations.npy")

    n_iter = rewards.shape[0]
    n_record = real_rewards.shape[0]
    record_period = n_iter//n_record
    slice_size = 500
    
    rewards = np.reshape(rewards, (n_iter//slice_size, slice_size)).mean(axis=1)
    iterations = np.reshape(iterations, (n_iter//slice_size, slice_size)).mean(axis=1)
    loss = np.reshape(loss, (n_iter//slice_size, slice_size)).mean(axis=1)

    x = list(range(0, n_iter, slice_size))
    xx = list(range(1, n_iter, record_period))
    plt.plot(x, rewards)
    plt.plot(xx, real_rewards, color='red')
    plt.show()
    plt.plot(x, iterations)
    plt.plot(xx, real_iterations, color='red')
    plt.show()
    # plt.plot(x, loss)
    # plt.show()