#! /usr/bin/env python

import numpy as np
import random as random
import matplotlib.pyplot as plt
import seaborn as sns

random.seed()
np.random.seed()

# Here, we consider slot machines as one-armed bandits.

class Environment :
    ''' Environment class '''


    def __init__(this, arms_parameters):
        '''constructor for a general environment
        takes 1 argument arms_parameters : list of probability parameters of each arm'''
        this.arms_parameters = arms_parameters
        this.arms_nb = len(arms_parameters)


    def getArmParameter(this, n):
        '''accessor of arm n parameters
        takes 1 argument n : arn number n'''
        return this.arms_parameters[n]

    def getNbArms(this):
        '''accessor of number of arms'''
        return this.arms_nb


class GaussianEnvironment(Environment):
    ''' GaussianEnvironment class where rewards for each arm are normally distributed
    takes 1 argument : arms_parameters list with the parameters (expectancy and variance) for each arm'''


    def __init__(this, arms_parameters):
        '''constructor for GaussianEnvironment'''
        super(GaussianEnvironment, this).__init__(arms_parameters)


    def whoAmI(this):
        '''prints a description of the working environment'''
        print(f"In this environment there are {this.arms_nb} bandit arms that give rewards following a gaussian distribution")


    def getReward(this, n, size=1):
        '''returns reward for arm n
        takes 2 arguments : n the number of the arm
                            size the number of draws'''
        reward = 0
        for i in range(size):
            reward += np.random.normal(this.arms_parameters[n][0], this.arms_parameters[n][1])
        return round(reward, 2)


    def showArmDistribution(this):
        '''shows (and saves) a plot of the distribution of rewards from each arm'''
        plt.figure(figsize=(10,5))
        pastels = sns.color_palette("pastel", this.arms_nb)
        dataList = []

        for i, p in enumerate(this.arms_parameters):
            dataList.append(np.random.normal(p[0],p[1],400))
            plt.text(i-0.2, p[0], f'µ= {round(p[0],2)}', size='6')

        ax = sns.violinplot(data=dataList, cut=0, palette=pastels, inner=None)

        # g.set_xticklabels([x for x in range(1,this.arms_nb+1)]) its ok if arm starts at 0
        plt.xlabel('Slot Machine Number')
        plt.ylabel('Rewards Distribution')
        plt.title('Rewards distribution for gaussian slot machines')
        #ax.tick_params(axis='x', which='major', pad=10)
        #ax.set_xticks([1+x for x in range(0,10)])
        plt.savefig('Gaussian/GaussianDistributions.png')

        plt.show()


class BernoulliEnvironment(Environment):
    ''' UniformEnvironment class where rewards for each arm are uniformy distributed
    takes 1 argument : arms_parameters list with the expectancies for each arm'''


    def __init__(this, arms_parameters):
        '''constructor for BernoulliEnvironment'''
        super(BernoulliEnvironment, this).__init__(arms_parameters)


    def whoAmI(this):
        '''prints a description of the working environment'''
        print(f"In this environment there are {this.arms_nb} bandit arms that give rewards following a bernoulli distribution")


    def getReward(this, n, size=1):
        '''takes 2 arguments : n the number of the arm
                            size the number of draws'''
        reward = 0
        for i in range(size):
            reward = np.random.binomial(1,this.arms_parameters[n])
        return reward


    def showArmDistribution(this):
        '''shows (and saves) a plot of the distribution of rewards from each arm'''
        plt.figure(figsize=(10,5))
        dataList = np.array([])

        for i in this.arms_parameters:
            dataList = np.append(dataList, [np.sum(np.random.binomial(1,i,size=100))])
        dataList = dataList/100

        plt.bar(np.arange(this.arms_nb), dataList, align='center', alpha=0.5, color= sns.color_palette("pastel", this.arms_nb))
        plt.xticks(np.arange(this.arms_nb))
        plt.xlabel('Slot Machine Number')
        plt.ylabel('Rewards Distribution')
        plt.title('Rewards distribution for bernoulli slot machines')

        plt.savefig('Bernoulli/BernoulliDistributions.png')

        plt.show()


class ConstantEnvironment(Environment):
    ''' ConstantEnvironment class where rewards for each arm are constants
    takes 1 argument : arms_parameters list with the constant reward for each arm'''


    def __init__(this, arms_parameters):
        '''constructor for BernoulliEnvironment'''
        super(ConstantEnvironment, this).__init__(arms_parameters)


    def whoAmI(this):
        '''prints a description of the working environment'''
        print(f"In this environment there are {this.arms_nb} bandit arms that give constant rewards")


    def getReward(this, n, size=1):
        '''takes 2 arguments : n the number of the arm
                            size the number of draws'''
        reward = 0
        for i in range(size):
            reward = this.arms_parameters[n]
        return reward


    def showArmDistribution(this):
        '''shows (and saves) a plot of the distribution of rewards from each arm'''
        plt.figure(figsize=(10,5))

        plt.scatter(np.arange(this.arms_nb), [p for p in this.arms_parameters], marker='o', linestyle='None', color= sns.color_palette("pastel", this.arms_nb))

        plt.xlabel('Slot Machine Number')
        plt.xticks(np.arange(this.arms_nb))
        plt.yticks(np.arange(max(this.arms_parameters)+1))
        plt.ylabel('Rewards Distribution')
        plt.title('Rewards distribution for constant slot machines')

        plt.savefig('Constant/ConstantDistributions.png')

        plt.show()
