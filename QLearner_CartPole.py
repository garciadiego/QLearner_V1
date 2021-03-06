import gym
import numpy as np
import tensorflow as tf



#mySoftmax Layer

def softmax(x):
    dg_exp_= np.exp(x - np.max(x))
    ans = dg_exp_ / dg_exp_.sum()
    return ans

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observ = env.reset()
    total_reward=0
    param =np.random.rand(4)*2-1

    for t in range(100):
        env.render()
        print(observ)

        action = 0 if np.matmul(param,observ) < 0 else 1

        observ, reward, done, info = env.step(action)
        total_reward += reward

        print ("Total reward is %d"). format (t+1) % total_reward

        if done:
            print ("Total reward is %d"). format (t+1) % total_reward
            print("Episode finished after {} timesteps".format(t+1))
            break
