# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:20:46 2017

@author: Noah_Huang
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

#Declare variables
learning_rate = 0.6
discont_rate = 0.99

num_of_episode = 2000

reward_list = []
action_list = []

#Create environment
env = gym.make ('FrozenLake-v0')

print ("number of observation:",env.observation_space.n, ", number of action:",env.action_space.n)

#Create Q-table
Q = np.zeros ([env.observation_space.n, env.action_space.n])

for _ in range (num_of_episode):
   s = env.reset ()
   total_reward = 0
   get_destination = False
   step = 0
   
   while step < 99:
       step += 1
       a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(step+1)))
       s1, reward, get_destination,_ = env.step(a)
       Q[s,a] = Q[s,a] + learning_rate * (reward + discont_rate * np.argmax(Q[s1,:]) - Q[s,a])
       total_reward += reward
       s = s1
       if get_destination == True:
           #print ("episode:",step," got des with reward:",total_reward)
           break
       
   reward_list.append (total_reward)
   
print ("Score over time: " +  str(sum(reward_list)/num_of_episode))
plt.plot (reward_list)
plt.show ()
   