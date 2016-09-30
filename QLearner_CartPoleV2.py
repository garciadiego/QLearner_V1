import tensorflow as tf
import numpy as np
import random
import gym

#Reinforcement Learning Method
def policy():
    with tf.variable_scope("policy"):
        parameters = tf.get_variable("policy_parameters",[4,2])
        #place holder for correct answers
        state = tf.placeholder(tf.float32,[None,4]) #dim any lenght x 4 (4 observations)
        actions = tf.placeholder(tf.float32,[None,2]) #dim any lenght x 2 (actions)
        advantages = tf.placeholder(tf.float32,[None,1]) #dim any lenght X 1 (Reward)
        y = tf.matmul(state,parameters)
        y_ = tf.nn.softmax(y)
        weights = tf.reduce_sum(tf.mul(y_, actions), reduction_indices=[1])
        eligibility = tf.log(weights) * advantages
        loss = -tf.reduce_sum(eligibility)
        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
        return y, state, actions, advantages, train_step


#Model
def model():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4]) #place holder for input
        newStates = tf.placeholder("float",[None,1]) #place holder for output

        #First Layer
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1) #activation function

        calculated_lay1 = tf.matmul(h1,w2) + b2
        calculated = calculated_lay1
        diffs = calculated - newStates
        loss = tf.nn.l2_loss(diffs**2)
        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
        return calculated, state, newStates,loss, train_step




def run_episode(env, policy, model, sess):
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []


    for _ in xrange(200):
        env.render()
        action = 0 if random.uniform(0,1) < [0][0] else 1 #draw samples from a uniform distribution
        #action = env.action_space.sample() #random action
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2) #array of zeros
        actionblank[action] = 1
        actions.append(actionblank)

        # take the action in the environment
        #old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((observation, action, reward))
        totalreward += reward
        if done:
            break

    return totalreward

#Enviroment Load
env = gym.make('CartPole-v0')
episodes_per_update = 5
#env.monitor.start('cartpole-hill/', force=True)

policy = policy()
model = model()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in xrange(1000):
    reward = run_episode(env, policy, model, sess)
    print "reward %d " % (reward)
    if reward == 100:
        break

#env.monitor.close()
