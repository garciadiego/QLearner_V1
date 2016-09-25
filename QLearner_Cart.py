
import gym
env = gym.make('MountainCar-v0')

for i_episode in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
	
		action = 2 if observation[0] < -1  or observation[1] > 0 else 0  
		observation, reward, done, info = env.step(action)
		if done:
			print "Episode %d finished after {} timesteps".format(t+1) % i_episode
			
			break
