import gym
import numpy as np
import tensorflow as tf

def weights_x(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_x(shape):
  initial = tf.zeros(shape, dtype=tf.float32, name=None)
  return tf.Variable(initial)

#Envairoment Load and Intial State
env = gym.make('CartPole-v0')
env.monitor.start('/Users/diegogarcia/Desktop/Deep_Learning/Qlearner_Main/QLearner_V1/cartpole-experiment-1', force=True)
dim_actions = env.action_space.n
num_gradients = 1
maxsteps = 1000
num_runs = 1000
sess = tf.InteractiveSession()

#Placeholders
state = tf.placeholder(tf.float32, shape=[None, 4])
action_choice = tf.placeholder(tf.float32, shape=[None, dim_actions])
reward_signal = tf.placeholder(tf.float32, shape=(None,1) )
n_timesteps = tf.placeholder(tf.float32, shape=())

#Layers
W1 = weights_x([4, 10])
b1 = bias_x([10])
h1 = tf.nn.relu(tf.matmul(state, W1) + b1)
W2 = weights_x([10, dim_actions])
b2 = bias_x([dim_actions])
h2 = tf.nn.softmax(tf.matmul(h1, W2) + b2)


tp = tf.transpose(action_choice)
log_prob = tf.log(tf.diag_part(tf.matmul(h2, tp)))
log_prob = tf.reshape(log_prob, (1,-1))
loss = tf.matmul(log_prob, reward_signal)
loss = -tf.reshape(loss, [-1])
train_step = tf.train.AdamOptimizer().minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


timestep_learning = np.zeros((num_runs,1))
for run in range(num_runs):

    states = np.zeros((maxsteps,4), dtype='float32')
    actions = np.zeros((maxsteps,dim_actions), dtype='float32')
    rewards = np.zeros((maxsteps,1), dtype='float32')
    timestep =0
    observation = env.reset()
    observation = np.reshape(observation,(1,4))
    done = False

    while not done and timestep < maxsteps:
        if run % 50 == 0:
            env.render()
        action_prob = sess.run(h2, feed_dict={state: observation})
        action = np.argmax(np.random.multinomial(1, action_prob[0]))
        new_observation, reward, done, info = env.step(action)

        states[timestep, :] = observation

        actions[timestep, action] = 1
        rewards[timestep, :] = reward
        timestep += 1

        observation[:] = new_observation

    states = states[:timestep, :]
    actions = actions[:timestep, :]
    rewards = rewards[:timestep,:]
    rewards[:, 0] = np.cumsum(rewards[::-1])[::-1]

    if run % 10 == 0:
        print (rewards)
    for i in range(num_gradients):
        sess.run(train_step, feed_dict={state: states, action_choice: actions, reward_signal: rewards, n_timesteps: timestep})
    timestep_learning[run] = timestep

env.monitor.close()
env.render(close=True)
