# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
          Dictionary includes a single feature that
          is the state,action pair. This feature doesn't
          permit generalization.
        """
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
          Dictionary includes a feature for this
          exact state, this exact action, as well
          as the x coordinate and y coordinate of
          the state.
        """
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def manDist(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here it's all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        Returns simple features for a basic reflex Pacman:
        - whether food will be eaten
        - how far away the next food is
        - whether a ghost collision is imminent
        - whether a ghost is one step away
        """

        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class AdvancedFeatureExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        """
        Returns features for a basic reflex Pacman:
        - whether food will be eaten
        - how far away the next food is
        - whether a ghost collision is imminent
        - whether a ghost is one step away scared
        - whether a ghost is one step away not-scared

        Weights: {
            '#-of-ghosts-1-step-away-and-not-scared': -539.3399358389521,
            'closest-food': -4.015270353339094,
            '#-of-ghosts-1-step-away-and-scared': 291.74489904846456,
            'bias': 206.5118324929459,
            'eats-food': 306.8870548768516
        }

        These weights show that it is very undesirable for there to be a ghost very close to Pacman while not scared,
        but that it is very desirable for there to be a ghost close that IS scared. Similarly, the farther the closest
        food is, the worse Pacman is off, and that a particular action causes Pacman to eat food is desirable. 
        """
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghostStates = state.getGhostStates()
        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        if not action == None:
            dx, dy = Actions.directionToVector(action)
        else:
            dx, dy = 0, 0
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away-and-scared"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and g.scaredTimer > 1 for g in ghostStates)

        features["#-of-ghosts-1-step-away-and-not-scared"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and not g.scaredTimer > 1 for g in ghostStates)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away-and-not-scared"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
