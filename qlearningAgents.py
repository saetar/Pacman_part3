# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here... (initialized to a util.Counter())"
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        qValues = [self.getQValue(state, a) for a in self.getLegalActions(state)]
        if len(qValues) == 0:
            return 0.0
        return max(qValues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

          Iterates through each action and compares running optimal QValue to
          current calculated QValue. If they are the same, we add the action to
          a list of possible actions. If there are no actions that are optimal,
          we return None. Otherwise, we make a random choice for all actions that
          are tied with the same optimal QValue.
        """
        actions = self.getLegalActions(state)
        optimalQValue = float('-inf')
        optimalActions = []
        for action in actions:
            newQValue = self.getQValue(state, action)
            if newQValue == optimalQValue:
                optimalActions.append(action)
            elif newQValue > optimalQValue:
                optimalActions = [action]
                optimalQValue = newQValue
        if len(optimalActions) == 0:
            return None
        return random.choice(optimalActions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)

          Flips an epsilon-weighted coin to determine a random action
          versus our calculated optimal action.
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon): # randomly choose an action
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf

          Uses an alpha-weighted average between lastly-seen reward
          and all rewards we've seen before. Updates self.qValues
          Counter object.
        """
        oldQValue = self.getQValue(state, action)
        vStar = self.computeValueFromQValues(nextState)
        sample = reward + (self.discount * vStar)
        newQValue = ((1 - self.alpha) * oldQValue) + (self.alpha * sample)
        self.qValues[(state, action)] = newQValue


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"


    """ For Q7: Change the default values of epsilon and alpha in the signature below
    so that on smallGrid, the qlearning agent wins at least 80% of the time.
    You can change gamma if you wish, but you don't need to. The "YOUR CODE HERE" is
    to mark the function, but you don't actually need to write new code- just change
    the values.
    """
    "*** YOUR CODE HERE ***"
    def __init__(self, epsilon=0.25,gamma=0.8,alpha=0.35, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
          Implements dot product as described above. Checks to make sure state is not terminal.
          If it is, return 0.0 because no action is viable.
        """
        if state == 'TERMINAL_STATE':
            return 0.0
        features = self.featExtractor.getFeatures(state, action)
        runningSum = 0.0
        for feature in features:
            runningSum += features[feature] * self.weights[feature]
        return runningSum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition.
           Implements an update by alpha-update based on reward gained
           and difference calculated by the overall Q values.
        """
        features = self.featExtractor.getFeatures(state, action)
        a = self.computeActionFromQValues(nextState)
        discounted = self.discount * self.getQValue(nextState, a)
        difference = reward + discounted - self.getQValue(state, action)
        for feature in features:
            oldWeight = self.weights[feature]
            self.weights[feature] = oldWeight + self.alpha * difference * features[feature]

    def final(self, state):
        "Called at the end of each game. saetar: used just to print weights after training"
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print self.weights
            pass
