#! /usr/bin/env python

import os
from environments import *
from agents import *

if not os.path.isdir("Gaussian"):
    os.makedirs("Gaussian")

if not os.path.isdir("Bernoulli"):
    os.makedirs("Bernoulli")

if not os.path.isdir("Constant"):
    os.makedirs("Constant")


number_of_arms = 8


"""Epsilon Greedy Agent in a Gaussian Environment"""
means = np.random.normal(0,1,size=10)
variances = [0.5]*number_of_arms

gauss_parameters = list(zip(means, variances))
gaussEnv = GaussianEnvironment(gauss_parameters)

gaussEnv.showArmDistribution()

"""EpsilonGreedyAgent with epsilon = 0.1"""
myAgent = EpsilonGreedyAgent(gaussEnv, 0.1)
myAgent.initHistory()

y=[]
for i in range(400):
    myAgent.update()
    y.append(myAgent.averageRewardSoFar())

#course plot
plt.plot(myAgent.history[:,1], label='\u03B5 = 0.1')
plt.ylabel('reward')
plt.xlabel('step')
plt.legend()
plt.savefig('Gaussian/GaussianEpsilonGreedyCourse.png')
plt.close()


plt.plot(y, label='\u03B5 = 0.1')
plt.ylabel('average reward')
plt.xlabel('step')


"""EpsilonGreedyAgent with epsilon = 0.01"""
myAgent = EpsilonGreedyAgent(gaussEnv, 0.01)
myAgent.initHistory()

y=[]
for i in range(400):
    myAgent.update()
    y.append(myAgent.averageRewardSoFar())

plt.plot(y, label='\u03B5 = 0.01')


"""EpsilonGreedyAgent with epsilon = 0."""
myAgent = EpsilonGreedyAgent(gaussEnv, 0.)
myAgent.initHistory()

y=[]
for i in range(400):
    myAgent.update()
    y.append(myAgent.averageRewardSoFar())

plt.plot(y, label='\u03B5 = 0')

plt.legend()

plt.savefig('Gaussian/GaussianEpsilonGreedyAverageReward.png')
plt.close()


"""Epsilon Greedy Agent in a Bernoulli Environment"""
bernEnv = BernoulliEnvironment(np.random.random(size=number_of_arms))
bernEnv.showArmDistribution()

"""EpsilonGreedyAgent with epsilon = 0.1"""
myAgent = EpsilonGreedyAgent(bernEnv, 0.1)
myAgent.initHistory()

y=[]
for i in range(400):
    myAgent.update()
    y.append(myAgent.averageRewardSoFar())

#course
plt.plot(myAgent.history[:,1], label='\u03B5 = 0.1')
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Bernoulli/BernoulliEpsilonGreedyCourse.png')
plt.close()

"""EpsilonGreedyAgent with epsilon = 0.01"""
myAgent = EpsilonGreedyAgent(bernEnv, 0.1)
myAgent.initHistory()

y=[]
for i in range(400):
    myAgent.update()
    y.append(myAgent.averageRewardSoFar())

#average reward
plt.plot(y, label='\u03B5 = 0.1')
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Bernoulli/BernoulliEpsilonGreedyAverageReward.png')
plt.close()


"""Epsilon Greedy Agent in a Constant Environment"""
constEnv = ConstantEnvironment(np.random.randint(6, size=number_of_arms))
constEnv.showArmDistribution()

myAgent = EpsilonGreedyAgent(constEnv, 0.1)
myAgent.initHistory()

y=[]
for i in range(100):
    myAgent.update()
    y.append(myAgent.averageRewardSoFar())

#course
plt.plot(myAgent.history[:,1], label='\u03B5 = 0.1')
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Constant/ConstantEpsilonGreedyCourse.png')
plt.close()

#average reward
plt.plot(y, label='\u03B5 = 0.1')
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Constant/ConstantEpsilonGreedyAverageReward.png')
plt.close()
