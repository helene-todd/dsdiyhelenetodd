#! /usr/bin/env python

import numpy as np
import random as random

class Agent :

    def __init__(this, environment):
        this.env = environment


    def getHistory(this):
        '''accessor of actions and rewards history so far'''
        return this.history


    def getCurrentState(this):
        '''accessor of the current state we are in'''
        return this.history[-1]


    def update(this):
        '''calls nextState to go to the next step and updates history with action and reward'''
        this.nextState()


    def averageRewardSoFar(this):
        return np.mean(this.history[:,1])


    def bestArmSoFar(this):
        '''returns best arm based on information from past actions'''
        bestMean = 0
        bestArm = this.getCurrentState()[0] # default value to avoid errors
        for u in this.history:
            # calculate mean reward for each action, i.e. mean reward of machine
            average = np.mean(this.history[np.where(this.history[:,0] == u[0])][:, 1])
            if average >= bestMean:
                bestMean = average
                bestArm = u[0]
        return int(bestArm)


class EpsilonGreedyAgent(Agent):

    def __init__(this, environment, epsilon):
        super(EpsilonGreedyAgent, this).__init__(environment)
        this.epsilon = epsilon


    def initHistory(this):
        '''initializes the history array and begins with the best estimated arm over 10 draws'''
        rewardsAnalysis = [this.env.getReward(i,size=10) for i in range(this.env.arms_nb)]
        first_arm = rewardsAnalysis.index(max(rewardsAnalysis))
        this.history = np.array([[first_arm, this.env.getReward(first_arm)]])


    def nextState(this):
        '''goes to next epsilon greedy state'''
        rand = random.random()

        if rand > this.epsilon: # greedy exploitation action
            arm_choice = this.bestArmSoFar()
            next_state = np.array([[arm_choice, this.env.getReward(arm_choice)]])
            this.history = np.concatenate((this.history, next_state))

        else: # exploration action
            # choice = np.where(this.arms == np.random.choice(this.arms))[0][0]
            arm_choice = random.randint(0, this.env.getNbArms()-1)
            next_state = np.array([[arm_choice, this.env.getReward(arm_choice)]])
            this.history = np.concatenate((this.history, next_state)) # add to action-reward history array
