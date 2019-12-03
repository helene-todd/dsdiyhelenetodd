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

"""Epsilon Greedy Agent in a Gaussian Environment"""
number_of_arms = 10
means = np.random.normal(0,1,size=10)
variances = [0.5]*number_of_arms

gauss_parameters = list(zip(means, variances))
gaussEnv = GaussianEnvironment(gauss_parameters)

gaussEnv.showArmDistribution()

epsilon = 0.1
myAgent1 = EpsilonGreedyAgent(gaussEnv, epsilon)
myAgent1.initHistory()

y=[]
for i in range(400):
    myAgent1.update()
    y.append(myAgent1.averageRewardSoFar())

#course
plt.plot(myAgent1.history[:,1])
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Gaussian/GaussianEpsilonGreedyCourse.png')
plt.close()

#average reward
plt.plot(y)
plt.ylabel('average reward')
plt.xlabel('step')
plt.savefig('Gaussian/GaussianEpsilonGreedyAverageReward.png')
plt.close()


"""Epsilon Greedy Agent in a Bernoulli Environment"""
bernEnv = BernoulliEnvironment(np.random.random(size=number_of_arms))
bernEnv.showArmDistribution()

epsilon = 0.1
myAgent2 = EpsilonGreedyAgent(bernEnv, epsilon)
myAgent2.initHistory()

y=[]
for i in range(400):
    myAgent2.update()
    y.append(myAgent2.averageRewardSoFar())

#course
plt.plot(myAgent2.history[:,1])
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Bernoulli/BernoulliEpsilonGreedyCourse.png')
plt.close()

#average reward
plt.plot(y)
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Bernoulli/BernoulliEpsilonGreedyAverageReward.png')
plt.close()


"""Epsilon Greedy Agent in a Constant Environment"""
constEnv = ConstantEnvironment(np.random.randint(6, size=number_of_arms))
constEnv.showArmDistribution()

epsilon = 0.1
myAgent3 = EpsilonGreedyAgent(constEnv, epsilon)
myAgent3.initHistory()

y=[]
for i in range(100):
    myAgent3.update()
    y.append(myAgent3.averageRewardSoFar())

#course
plt.plot(myAgent3.history[:,1])
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Constant/ConstantEpsilonGreedyCourse.png')
plt.close()

#average reward
plt.plot(y)
plt.ylabel('reward')
plt.xlabel('step')
plt.savefig('Constant/ConstantEpsilonGreedyAverageReward.png')
plt.close()


"""
        def plotCourse(): (maybe in subClass)
        for u in history:
            runningMean = np.mean(history[np.where(history[:,0] == u[0])][:, 1])
        #calculate the mean reward
        runningMean = np.mean(this.history[:,1])
        x.append(i)
        y.append(runningMean)

    def plotCourse():

    #plt.scatter(i, runningMean)
"""
