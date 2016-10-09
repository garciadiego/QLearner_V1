import gym
import numpy as np
import tensorflow as tf
import math
import random

# HYPER_PARAMETERS
hidden_layer1 = 50  #number of hidden layers on 1
hidden_layer2 = 50 #number of hidden layers on 2

batch_number = 500
gamma = 0.9995
num_of_ticks_between_q_copies = 1000
explore_decay = 0.99995
min_explore = 0.05
max_steps = 199
max_episodes = 1000
memory_size = 20000
learning_rate = 1e-3

#setup enviroment
env = gym.make('CartPole-v0')
#creates a monitor for the training
env.monitor.start('/tmp/QLearner/cartpole', force=True)
tf.reset_default_graph()

#Q Network
#observation initial states
observation_init = tf.random_uniform([env.observation_space.shape[0],hidden_layer1], -0.10, 0.10)
observation_init_Prime = tf.random_uniform([env.observation_space.shape[0],hidden_layer1], -1.0, 1.0)

w1 = tf.Variable(observation_init)
b1 = tf.Variable(tf.random_uniform([hidden_layer1], -0.10, 0.10))

w2 = tf.Variable(tf.random_uniform([hidden_layer1, hidden_layer2], -0.10, 0.10))
b2 = tf.Variable(tf.random_uniform([hidden_layer2], -0.10, 0.10))

w3 = tf.Variable(tf.random_uniform([hidden_layer2, env.action_space.n], -0.10, 0.10))
b3 = tf.Variable(tf.random_uniform([env.action_space.n], -0.10, 0.10))

w1_prime = tf.Variable(observation_init_Prime)
b1_prime = tf.Variable(tf.random_uniform([hidden_layer1], -1.0, 1.0))

w2_prime = tf.Variable(tf.random_uniform([hidden_layer1,hidden_layer2],-1.0, 1.0))
b2_prime = tf.Variable(tf.random_uniform([hidden_layer2], -1.0, 1.0))

w3_prime = tf.Variable(tf.random_uniform([hidden_layer2, env.action_space.n], -1.0))
b3_prime = tf.Variable(tf.random_uniform([env.action_space.n], -1.0))

#Assigns Q prime's weights updates
w1_prime_update= w1_prime.assign(w1)
b1_prime_update= b1_prime.assign(b1)

w2_prime_update= w2_prime.assign(w2)
b2_prime_update= b2_prime.assign(b2)

w3_prime_update= w3_prime.assign(w3)
b3_prime_update= b3_prime.assign(b3)

#lists of assigns
all_assigns = [w1_prime_update, w2_prime_update, w3_prime_update,
                b1_prime_update, b2_prime_update, b3_prime_update]


#Hidden Network Set up (Activation Functions)
states_placeholder = tf.placeholder(tf.float32, [None, env.observation_space.shape[0]])
hidden_activationfx_1 = tf.nn.relu(tf.matmul(states_placeholder, w1) + b1)
hidden_activationfx_2 = tf.nn.relu(tf.matmul(hidden_activationfx_1, w2) + b2)
#weights dropout
#hidden_activationfx_2 = tf.nn.dropout(hidden_activationfx_2, .5)

#Q-Value
Q = tf.matmul(hidden_activationfx_2, w3) + b3

#Prime Hidden Network Set up
hidden_activationfx_1_prime = tf.nn.relu(tf.matmul(states_placeholder, w1_prime) + b1_prime)
hidden_activationfx_2_prime = tf.nn.relu(tf.matmul(hidden_activationfx_1_prime, w2_prime) + b2_prime)
#Prime Layer weights dropout (Saw it in last class)
#hidden_activationfx_2_prime = tf.nn.dropout(hidden_activationfx_2_prime, .5)

#Q Prime Value
Q_prime =  tf.matmul(hidden_activationfx_2_prime, w3_prime) + b3_prime

action_used_placeholder = tf.placeholder(tf.int32, [None], name="action_masks")

#tf one hot vector (Change value by 1)
action_masks = tf.one_hot(action_used_placeholder, env.action_space.n)

filtered_Q = tf.reduce_sum(tf.mul(Q, action_masks), reduction_indices=1)

#Q training with adam optimizer
target_q_placeholder = tf.placeholder(tf.float32, [None])
loss = tf.reduce_sum(tf.square(filtered_Q - target_q_placeholder))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

##############ENVIROMENT LOAD AND SET UP################
Experience_Replay = [] #we saved our experiance replay here
explore = 1.0   #Epsilon Greedy
rewardList = []
past_actions = []
episode_number = 0
episode_reward = 0

xmax = 1
ymax = 1
xind = 1
yind = 3

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    #Copy Q over to Q_prime
    sess.run(all_assigns)
    ticks = 0

    for episode in xrange(max_episodes):
        state = env.reset()
        teststate = state
        reward_sum = 0

        for step in xrange(max_steps):
            #ticks += 1
            #print state
            #xmax = max(xmax, state[xind])
            #ymax = max(ymax, state[yind])

            if episode % 15 == 0:
                q, qp = sess.run([Q,Q_prime], feed_dict={states_placeholder: np.array([state])})
                print "Q:{}, Q_Prime {}".format(q[0], qp[0])
                #print "T: {} S {}".format(ticks, state)
                #teststate = state
                env.render()

            #if explore > np.random.rand(1):
            if explore > random.random():
                action = env.action_space.sample()
            else:
                #get action from policy
                q = sess.run(Q, feed_dict={states_placeholder: np.array([state])})[0]
                action = np.argmax(q)
                #print action
            explore = max(explore * explore_decay, min_explore)

            new_state, reward, done, _ = env.step(action)
            reward_sum += reward
            Experience_Replay.append([state, action, reward, new_state, done])

            #Memory System
            if len(Experience_Replay) > memory_size:
                Experience_Replay.pop(0); #takes the first value and deleted from the list
            state = new_state

            if done:
                break

            #Training Batch
            samples = random.sample(Experience_Replay, min(batch_number, len(Experience_Replay)))

            #print samples

            #calculate all next Q's together for speed
            new_states = [ x[3] for x in samples]
            all_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: new_states})

            y_ = []
            state_samples = []
            actions = []
            terminalcount = 0
            for ind, i_sample in enumerate(samples):
                state_mem, curr_action, reward, new_state, done = i_sample
                if done:
                    y_.append(reward)
                    terminalcount += 1
                else:
                    #this_q_prime = sess.run(Q_prime, feed_dict={states_placeholder: [new_state]})[0]
                    this_q_prime = all_q_prime[ind]
                    maxq = max(this_q_prime)
                    y_.append(reward + (gamma * maxq))

                state_samples.append(state_mem)

                actions.append(curr_action);
            sess.run([train], feed_dict={states_placeholder: state_samples, target_q_placeholder: y_, action_used_placeholder: actions})
            if ticks % num_of_ticks_between_q_copies == 0:
                sess.run(all_assigns)


        print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)
env.monitor.close()
