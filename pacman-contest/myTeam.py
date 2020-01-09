# -*- coding:utf-8 -*-


'''

                              _____                                   _  ___
                             |  __ \                                 | |/ (_)
                             | |__) |_ _  ___ _ __ ___   __ _ _ __   | ' / _ _ __   __ _
                             |  ___/ _` |/ __| '_ ` _ \ / _` | '_ \  |  < | | '_ \ / _` |
                             | |  | (_| | (__| | | | | | (_| | | | | | . \| | | | | (_| |
                             |_|   \__,_|\___|_| |_| |_|\__,_|_| |_| |_|\_\_|_| |_|\__, |
                                                                                    __/ |
                                                                                   |___/

                                              ---      ----      ---
                                              |  \    /  + \    /  |
                                              | + \--/      \--/ + |
                                              |   +     +          |
                                              | +     +        +   |
                                            @@@@@@@@@@@@@@@@@@@@@@@@@@
                                          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                                  @@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                                    @@@@@@@@@@@@@@@@@@@@@@@@@@
                                                      @@@@@@@@@@@@@@@@@@@@@@@@
                                                        @@@@@@@@@@@@@@@@@@@@@@
                                                        @@@@@@@@@@@@@@@@@@@@@@
                                                     @@@@@@@@@@@@@@@@@@@@@@@@@
                                                   @@@@@@@@@@@@@@@@@@@@@@@@@@@
                                              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                             @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                                            @@@@@@@@@@@@@@@@@@@@@@@@@@
                                                @@@@@@@@@@@@@@@@@@

'''

import random
import util
import time
import collections
from game import Directions
from game import Actions
from util import PriorityQueue
from captureAgents import CaptureAgent
from util import Queue
import numpy as np
from util import nearestPoint
from baselineTeam import ReflexCaptureAgent
from baselineTeam import DefensiveReflexAgent
import operator
import sys

sys.path.append('teams/RLNB2')

GLOBAL_DEBUG = False
DEBUG = False
PRINT_STEP_TIME = False

CARRY_THRESHOLD = 4
TIME_THRESHOLD = 150


firstAgent = 'MyRuleOffensiveAgent'
secondAgent = 'MyRuleOffensiveAgent'


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first=firstAgent, second=secondAgent):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class MyRuleOffensiveAgent(CaptureAgent):
    __hmmAgent = None

    @classmethod
    def initHmmAgent(cls, gameState, opponentsIndeces):
        if cls.__hmmAgent == None:
            cls.__hmmAgent = HmmAgent(gameState, opponentsIndeces)

    @classmethod
    def getHmmAgent(cls, gameState, opponentsIndeces):
        if cls.__hmmAgent == None:
            cls.__hmmAgent = HmmAgent(gameState, opponentsIndeces)
        else:
            return cls.__hmmAgent

    #################################################################################
    # Overwritten Agent Core Functions #
    #################################################################################
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.homeBorderPosList = getHomeBorderPos(gameState, self.red)
        self.enemyBorderPosList = getEnemyBorderPos(gameState, self.red)
        self.enemies = self.getOpponents(gameState)
        self.initialFoods = len(self.getFood(gameState).asList())
        self.foodsDelivered = 0
        self.foodsCarrying = gameState.getAgentState(self.index).numCarrying
        self.lastState = gameState
        self.lastAction = None
        self.isFirstAgent = self.isFirstCheck(gameState)
        self.middleBorder = getMiddleBorder(gameState)
        self.lowerHalfFoods, self.higherHalfFoods = self.getLowerAndHigherFoods(gameState)
        self.currentTarget = None
        self.cornerAndCorridorDict = getCornerAndCorridorDict(gameState, self.red)
        self.statesQueue = Queue()
        self.Corner = Corner(gameState, self.red)
        self.opponentsIndeces = self.getOpponents(gameState)
        self.foodDefend = self.getFoodAndCapsuleDefend(gameState)
        self.initHmmAgent(gameState, self.opponentsIndeces)
        self.turnBack = False

        if GLOBAL_DEBUG:
            self.printGlobalInfo(gameState)

    def chooseAction(self, gameState):
        start = time.time()

        hmmAgent = self.getHmmAgent(gameState, self.opponentsIndeces)
        hmmAgent.updateObservationList(self, gameState)
        self.foodDefend = self.getFoodAndCapsuleDefend(gameState)

        self.updateGlobalInfo(gameState)
        if DEBUG:
            self.printDebugInfo(gameState)

        # if not pacman, join the battle, else eat food or go home
        if self.isRepeating(gameState):
            # print("agent {} is rdepeating ".format(self.index))
            legalActions = gameState.getLegalActions(agentIndex=self.index)
            my_random = random.Random()
            my_random.seed(time.time())
            final_action = my_random.choice(legalActions)

        elif self.hasToGoDefense(gameState):
            final_action = self.defensiveAction(gameState)
        else:
            if not isPacman(gameState, self.index):
                final_action = self.eatFoodAction(gameState)
            else:
                if self.isBeingChasing(gameState):
                    # print("is being chasing")
                    final_action = self.beingChasingAction(gameState)
                elif (len(self.getFoodsLeft(gameState)) <= 2) or \
                        (gameState.data.timeleft < TIME_THRESHOLD and self.foodsCarrying > 0) or \
                        (0 < len(self.shortestPathToHome(gameState)) <= 5 and self.foodsCarrying >= CARRY_THRESHOLD
                        and not self.allEnemyIsPacman(gameState)):
                    final_action = self.goHomeAction(gameState)
                else:
                    final_action = self.eatFoodAction(gameState)

        if PRINT_STEP_TIME:
            printExecutionTime(start)

        self.lastState = gameState
        self.lastAction = final_action
        return final_action

    #################################################################################
    # Rule Functions #
    #################################################################################
    def hasToGoDefense(self, gameState):
        # global SomeOneIsDefensing
        # if some one else is defensing , I don't have to go defense
        if len(self.getFoodsLeft(gameState)) <= 2 and self.foodsCarrying == 0:
            return True

        if getScaredTime(gameState, self.index) > 0:
            return False

        self_pos = getPosition(gameState, self.index)
        opponent_pacmans = self.observedEnemyPacman(gameState)

        team_mate_ID = self.getTeamMateID(gameState)
        team_mate_pos = getPosition(gameState, team_mate_ID)

        my_total_distance = 0
        team_mate_total_distance = 0

        for oppo_pacman in opponent_pacmans:
            oppo_pacman_position = getPosition(gameState, oppo_pacman)
            my_total_distance += self.getMazeDistance(self_pos, oppo_pacman_position)
            team_mate_total_distance += self.getMazeDistance(team_mate_pos, oppo_pacman_position)

        # if I am nearer, I go defense, or my team mate go defense
        if my_total_distance < team_mate_total_distance:
            return True
        else:
            return False

    def defensiveAction(self, gameState):
        # self.debugClear()
        # print("agent {0} defensive action".format(self.index))

        current_pos = getPosition(gameState, self.index)
        all_positions_avoid = self.dangerousPositions(gameState)
        oppo_pacman = []
        oppo_pacman_positions = []
        for enemy in self.enemies:
            position = self.getHmmAgent(gameState, self.opponentsIndeces).getApproximateOpponentPos(enemy)
            if isPacman(gameState, enemy):
                oppo_pacman.append(enemy)
                oppo_pacman_positions.append(position)
        # print("oppo pacman pos: ", oppo_pacman_positions)
        # self.debugDraw(cells=oppo_pacman_positions, color=[1, 0, 0])

        shortest_path = None
        shortest_length = np.inf
        for position in oppo_pacman_positions:
            path = self.aStarSearch(gameState, current_pos, [position], avoidPositions=all_positions_avoid)
            if path is None:
                continue
            if len(path) < shortest_length:
                shortest_length = len(path)
                shortest_path = path
        if shortest_path is None or len(shortest_path) == 0:
            return self.struggleAction(gameState)

        return shortest_path[0]

    def eatFoodAction(self, gameState):
        CALCULATE_LIMIT = 15
        # print("agent {0} eat food action".format(self.index))
        all_positions_avoid = self.dangerousPositions(gameState)
        current_pos = getPosition(gameState, self.index)

        if self.turnBack:
            # print("turning back")
            return self.goHomeAction(gameState)

        all_eatable = []
        foods = self.getFoodsLeft(gameState)
        all_eatable.extend(foods)

        my_all_eatable = []
        if len(foods) > CALCULATE_LIMIT:
            mazeDistForFood = dict()
            for food in all_eatable:
                mazeDistForFood.update({
                    food: self.getMazeDistance(current_pos, food)
                })
            sorted_foods_dict = sorted(mazeDistForFood.items(), key=operator.itemgetter(1))

            sorted_foods = []
            for food in sorted_foods_dict:
                sorted_foods.append(food[0])
            my_all_eatable = sorted_foods[0: CALCULATE_LIMIT]
        else:
            my_all_eatable = all_eatable

        all_paths = dict()
        for eat in my_all_eatable:
            path = self.aStarSearch(gameState, current_pos, [eat], avoidPositions=all_positions_avoid)
            if path is None:
                continue
            all_paths.update({
                eat: len(path)
            })
        if len(all_paths) == 0:
            return self.goHomeAction(gameState)

        sorted_eatable = sorted(all_paths.items(), key=operator.itemgetter(1))
        nearest = sorted_eatable[0][0]
        nearest_path_len = sorted_eatable[0][1]


        mate_all_paths = dict()
        mate_pos = getPosition(gameState, self.getTeamMateID(gameState))
        mate_all_eatable = []
        if len(foods) > CALCULATE_LIMIT:
            mazeDistForFood = dict()
            for food in all_eatable:
                mazeDistForFood.update({
                    food: self.getMazeDistance(mate_pos, food)
                })
            sorted_foods_dict = sorted(mazeDistForFood.items(), key=operator.itemgetter(1))

            sorted_foods = []
            for food in sorted_foods_dict:
                sorted_foods.append(food[0])
            mate_all_eatable = sorted_foods[0: CALCULATE_LIMIT]
        else:
            mate_all_eatable = all_eatable

        for eat in mate_all_eatable:
            path = self.aStarSearch(gameState, mate_pos, [eat], avoidPositions=all_positions_avoid)
            if path is None:
                continue
            mate_all_paths.update({
                eat: len(path)
            })
        if len(mate_all_paths) != 0:
            mate_sorted_eatable = sorted(mate_all_paths.items(), key=operator.itemgetter(1))
            mate_nearest = mate_sorted_eatable[0][0]
            mate_nearest_path_len = mate_sorted_eatable[0][1]

            # choose second nearest
            if nearest == mate_nearest and nearest_path_len > mate_nearest_path_len and len(all_paths) >= 2:
                nearest = sorted_eatable[1][0]

            if len(all_eatable) <= 4 and nearest_path_len > mate_nearest_path_len:
                if self.foodsCarrying > 0:
                    return self.goHomeAction(gameState)
                else:
                    return self.defensiveAction(gameState)

        final_path = self.aStarSearch(gameState, current_pos, [nearest], avoidPositions=all_positions_avoid)

        return final_path[0]

    def goHomeAction(self, gameState):
        # print("agent {0} go home action".format(self.index))
        shortest_path = self.shortestPathToHome(gameState)

        if len(shortest_path) == 0:
            return self.struggleAction(gameState)
        return shortest_path[0]

    # if is being chasing, if any capsule available, go and get that capsule
    def beingChasingAction(self, gameState):
        # print("agent {0} being chasing action".format(self.index))

        shortest_path = self.shortestPathToHome(gameState)
        if len(shortest_path) > 0 and ((len(self.getFoodsLeft(gameState)) <= 2) or
                (gameState.data.timeleft < TIME_THRESHOLD and self.foodsCarrying > 0) or
                (len(shortest_path) <= 5 and self.foodsCarrying >= CARRY_THRESHOLD
                 and not self.allEnemyIsPacman(gameState))):
            return shortest_path[0]

        mostDangerous = self.mostDangerousPositions(gameState)
        myPos = getPosition(gameState, self.index)
        capsules = self.getCapsules(gameState)
        nearestCapsule = None
        nearestDistance = 10000  # true path distance
        for capsule in capsules:
            path = self.aStarSearch(gameState, myPos, [capsule], mostDangerous)
            if path is None:
                continue
            if len(path) < nearestDistance:
                nearestDistance = len(path)
                nearestCapsule = capsule
        if nearestCapsule is not None:
            pathToNearestCapsule = self.aStarSearch(gameState, myPos, [nearestCapsule], mostDangerous)
            return pathToNearestCapsule[0]
        else:
            if len(shortest_path) == 0:
                return self.struggleAction(gameState)
            else:
                return shortest_path[0]

    def struggleAction(self, gameState):
        # print("struggle action")
        # path_to_home = self.shortestPathToHomeWhenStruggle(gameState)
        # if path_to_home is not None and len(path_to_home) != 0:
        #     return path_to_home[0]
        # else:  # ghosts on my way, find a possible action to go far away of ghost
        final_action = self.moveFarAwayFromGhostAction(gameState)
        return final_action


    def moveFarAwayFromGhostAction(self, gameState):
        legalActions = gameState.getLegalActions(agentIndex=self.index)
        allEnemy = self.observedEnemy(gameState)
        myPos = getPosition(gameState, self.index)
        nearestRealGhost = None
        nearestDistance = 10000
        for enemy in allEnemy:
            # get away from it
            if not isPacman(gameState, enemy) and getScaredTime(gameState, enemy) == 0:
                distance = self.getMazeDistance(myPos, getPosition(gameState, enemy))
                if distance < nearestDistance:
                    nearestDistance = distance
                    nearestRealGhost = enemy

        if nearestRealGhost is None:
            return Directions.STOP

        for action in legalActions:
            successor = self.getSuccessor(gameState, action)
            newPos = getPosition(successor, self.index)
            newDistance = self.getMazeDistance(newPos, getPosition(gameState, nearestRealGhost))
            if newDistance > nearestDistance and self.getMazeDistance(newPos, myPos) == 1 and \
                    not self.inCornerCorridor(successor):
                return action

        return Directions.STOP

    def hasToTurnBack(self, gameState):
        if not self.inCornerCorridor(gameState):
            return False

        if self.turnBack is True:
            return True

        if self.inCornerCorridor(gameState):
            current_pos = getPosition(gameState, self.index)

            gate_position = self.Corner.getGatePosition(current_pos)
            if gate_position is None:
                return False

            nearesrGhost = None
            nearestDistance = 10000
            for ghost in self.observedEnemyGhosts(gameState):
                ghost_pos = getPosition(gameState, ghost)
                distance = self.getMazeDistance(ghost_pos, gate_position)
                if distance < nearestDistance:
                    nearestDistance = distance
                    nearesrGhost = ghost

            distance_to_gate = self.getMazeDistance(current_pos, gate_position)
            # print("my distance to gate: ", distance_to_gate)
            # print("ghost distance to gate: ", nearestDistance)
            if distance_to_gate + 3 > nearestDistance:
                return True

        return False

    def inCornerCorridor(self, gameState):
        all_corner_and_corridor = []
        all_corner_and_corridor.extend(list(self.cornerAndCorridorDict['opponents']['cornerSet']))
        all_corner_and_corridor.extend(list(self.cornerAndCorridorDict['opponents']['corridorSet']))
        current_pos = getPosition(gameState, self.index)
        if current_pos in all_corner_and_corridor:
            return True
        else:
            return False

    ###########################################################################
    # Heuristic Search Functions #
    ###########################################################################

    def aStarSearch(self, gameState, startPosition, goalPositions, avoidPositions=[]):
        sourceNode = (startPosition, [], float('inf'))
        closedSet = set()
        openSet = util.PriorityQueue()
        openSet.push(sourceNode, 0)
        bestG = collections.defaultdict(lambda: float('inf'))

        if goalPositions[0] in avoidPositions:
            return None

        while not openSet.isEmpty():
            currentNode = openSet.pop()
            currentPosition = currentNode[0]
            currentActions = currentNode[1]
            currentGScore = currentNode[2]

            if currentPosition not in closedSet or currentGScore < bestG[currentPosition]:
                closedSet.add(currentPosition)
                bestG[currentPosition] = currentGScore

                if currentPosition in goalPositions:
                    return currentActions

                successors = self.getPossibleSuccessors(gameState, currentPosition, avoidPositions)
                for nextPosition, action in successors:
                    gScore = len(currentActions) + 1
                    hScore = max(self.getMazeDistance(nextPosition, goal) for goal in goalPositions)
                    fScore = gScore + hScore
                    nextNode = (nextPosition, currentActions + [action], gScore)
                    openSet.update(nextNode, fScore)

    def getPossibleSuccessors(self, gameState, currentPosition, avoidPositions):
        successors = []
        walls = gameState.getWalls()
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        for action in actions:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not walls[nextx][nexty] and (nextx, nexty) not in avoidPositions:
                successors.append(((nextx, nexty), action))
        return successors

    ###########################################################################
    # Helper Functions For Agents #
    ###########################################################################

    def updateGlobalInfo(self, gameState):
        self.turnBack = self.hasToTurnBack(gameState)
        # print("has to turn back", self.turnBack)
        if len(self.statesQueue.list) == 10:
            self.statesQueue.pop()
        self.statesQueue.push(gameState)

        self.foodsCarrying = gameState.getAgentState(self.index).numCarrying




    def hasDied(self, gameState):
        if self.getMazeDistance(getPosition(gameState, self.index),
                                getPosition(self.lastState, self.index)) > 1:
            return True
        else:
            return False

    def printGlobalInfo(self, gameState):
        print("\n*****************GLOBAL INFO******************")
        print("num agents, ", gameState.getNumAgents())
        print("my agents, ", self.getTeam(gameState))
        print("current minimax agent, ", self.index)
        print("enmey agents, ", self.enemies)
        print("self borders, ", self.homeBorderPosList)
        print("enemy borders, ", self.enemyBorderPosList)
        print("*****************GLOBAL INFO******************\n")

        home_cornerList = list(self.cornerAndCorridorDict['home']['cornerSet'])
        self.debugDraw(home_cornerList, [1, 0, 0])
        home_corridorList = list(self.cornerAndCorridorDict['home']['corridorSet'])
        self.debugDraw(home_corridorList, [0, 1, 0])
        opponents_cornerList = list(self.cornerAndCorridorDict['opponents']['cornerSet'])
        self.debugDraw(opponents_cornerList, [0, 0, 1])
        opponents_corridorList = list(self.cornerAndCorridorDict['opponents']['corridorSet'])
        self.debugDraw(opponents_corridorList, [1, 1, 1])

    def printDebugInfo(self, gameState):
        print("\n*****************STATE INFO******************")
        # currentObservation = self.getCurrentObservation()
        print("my agent 0 location: ", gameState.getAgentPosition(self.getTeam(gameState)[0]))
        print("my agent 1 location: ", gameState.getAgentPosition(self.getTeam(gameState)[1]))
        print("observed enemies: ", self.observedEnemy(gameState))
        for agent in self.getTeam(gameState):
            if isPacman(gameState, agent):
                print("friend {0} is pacman".format(agent))
            else:
                print("friend {0} is ghost".format(agent))
        for agent in self.getOpponents(gameState):
            if isPacman(gameState, agent):
                print("enemy {0} is pacman".format(agent))
            else:
                print("enemy {0} is ghost".format(agent))
        print("current agent: ", self.index)
        print("agent distances: ", gameState.getAgentDistances())
        print("time left: ", gameState.data.timeleft)
        print("*****************STATE INFO******************")

    def getFoodAndCapsuleDefend(self, gameState):
        return self.getFoodYouAreDefending(gameState).asList() + \
               self.getCapsulesYouAreDefending(gameState)

    def canObserveEnemy(self, gameState):
        if len(self.observedEnemy(gameState)) != 0:
            return True
        else:
            return False

    def allEnemyIsPacman(self, gameState):
        if isPacman(gameState, self.enemies[0]) and isPacman(gameState, self.enemies[1]):
            return True
        else:
            return False

    def observedEnemy(self, gameState):
        observedEnemy = []
        enemyIndex = self.getOpponents(gameState)
        if gameState.getAgentPosition(enemyIndex[0]) is not None:
            observedEnemy.append(enemyIndex[0])
        if gameState.getAgentPosition(enemyIndex[1]) is not None:
            observedEnemy.append(enemyIndex[1])

        return observedEnemy

    def observedEnemyGhosts(self, gameState):
        observedEnemy = self.observedEnemy(gameState)
        observed_ghost = []
        for enemy in observedEnemy:
            if not isPacman(gameState, agentIndex=enemy) and getScaredTime(gameState, enemy) <= 5:
                observed_ghost.append(enemy)
        return observed_ghost

    def observedEnemyPacman(self, gameState):
        observedEnemy = self.observedEnemy(gameState)
        observed_pacman = []
        for enemy in observedEnemy:
            if isPacman(gameState, agentIndex=enemy):
                observed_pacman.append(enemy)
        return observed_pacman

    def currentGhostPositions(self, gameState):
        ghosts = self.observedEnemyGhosts(gameState)
        ghosts_pos = []
        for ghost in ghosts:
            pos = getPosition(gameState, ghost)
            ghosts_pos.append(pos)

        return ghosts_pos

    def currentEnemyPacmanPositions(self, gameState):
        pacmans = self.observedEnemyPacman(gameState)
        pacman_pos = []
        for pacman in pacmans:
            pos = getPosition(gameState, pacman)
            pacman_pos.append(pos)
        return pacman_pos

    # current dangerous positions
    def dangerousPositions(self, gameState):
        current_pos = getPosition(gameState, self.index)
        all_pos = set()
        currentGhostsPositions = self.currentGhostPositions(gameState)
        all_pos.update(currentGhostsPositions)
        for ghostPos in currentGhostsPositions:
            enemyGhostNeighbors = getNeighbors(gameState, ghostPos)
            all_pos.update(enemyGhostNeighbors)

            if ghostPos in self.enemyBorderPosList:  # if enemy at their border, remove positions in our borders
                for neighbor in enemyGhostNeighbors:
                    if neighbor in self.homeBorderPosList and getScaredTime(gameState, self.index) == 0:
                        all_pos.remove(neighbor)

            if self.getMazeDistance(ghostPos, current_pos) <= 4:
                all_pos.update(self.cornerAndCorridorDict['opponents']['cornerSet'])
                all_pos.update(self.cornerAndCorridorDict['opponents']['corridorSet'])

        pacmanPositions = self.currentEnemyPacmanPositions(gameState)
        for position in pacmanPositions:
            if self.red and position in self.homeBorderPosList:
                neighborPosition = (int(position[0] + 1), int(position[1]))
                all_pos.add(neighborPosition)
            if not self.red and position in self.homeBorderPosList:
                neighborPosition = (int(position[0] - 1), int(position[1]))
                all_pos.add(neighborPosition)

            if self.getMazeDistance(position, current_pos) <= 4:
                all_pos.update(self.cornerAndCorridorDict['opponents']['cornerSet'])
                all_pos.update(self.cornerAndCorridorDict['opponents']['corridorSet'])

        if getScaredTime(gameState, self.index) > 0:
            all_pos.update(pacmanPositions)
            for position in pacmanPositions:
                all_pos.update(getNeighbors(gameState, position))

        return list(all_pos)

    def mostDangerousPositions(self, gameState):
        current_pos = getPosition(gameState, self.index)
        all_pos = set()
        currentGhostsPositions = self.currentGhostPositions(gameState)
        all_pos.update(currentGhostsPositions)
        for ghostPos in currentGhostsPositions:
            enemyGhostNeighbors = getNeighbors(gameState, ghostPos)
            all_pos.update(enemyGhostNeighbors)

            if ghostPos in self.enemyBorderPosList:  # if enemy at their border, remove positions in our borders
                for neighbor in enemyGhostNeighbors:
                    if neighbor in self.homeBorderPosList and getScaredTime(gameState, self.index) == 0:
                        all_pos.remove(neighbor)

        pacmanPositions = self.currentEnemyPacmanPositions(gameState)
        for position in pacmanPositions:
            if self.red and position in self.homeBorderPosList:
                neighborPosition = (int(position[0] + 1), int(position[1]))
                all_pos.add(neighborPosition)
            if not self.red and position in self.homeBorderPosList:
                neighborPosition = (int(position[0] - 1), int(position[1]))
                all_pos.add(neighborPosition)

        if getScaredTime(gameState, self.index) > 0:
            all_pos.update(pacmanPositions)
            for position in pacmanPositions:
                all_pos.update(getNeighbors(gameState, position))

        return list(all_pos)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        # pos = successor.getAgentState(self.index).getPosition()
        # if pos != nearestPoint(pos):
        #     # Only half a grid position was covered
        #     return successor.generateSuccessor(self.index, action)
        # else:
        return successor

    def getOpponentsCapsule(self, gameState):
        if self.red:
            return gameState.getRedCapsules()
        else:
            return gameState.getBlueCapsules()

    def getNearestBorderDistanceAndPos(self, gameState):
        # Compute the nearest position from current position in the border line
        myPos = gameState.getAgentPosition(self.index)
        bestDistance, bestIndex = self.getNearestPos(myPos, self.homeBorderPosList)
        return bestDistance, self.homeBorderPosList[bestIndex]

    def getNearestPos(self, pos, candidatePosList):
        if 0 == len(candidatePosList):
            return 0, 0
        bestDistance = float('inf')
        distanceList = [None] * len(candidatePosList)
        for index, candidatePos in enumerate(candidatePosList):
            dist = self.getMazeDistance(pos, candidatePos)
            distanceList[index] = dist
            if dist < bestDistance:
                bestDistance = dist
        bestIndexList = [index for index, value in enumerate(distanceList) if value == bestDistance]
        return bestDistance, random.choice(bestIndexList)

    def getFoodsLeft(self, gameState):
        return self.getFood(gameState).asList()

    def isFirstCheck(self, gameState):
        my_team = self.getTeam(gameState)
        if self.index == my_team[0]:
            return True
        else:
            return False

    def getLowerAndHigherFoods(self, gameState):
        # lower and higher half foods
        foodList = self.getFood(gameState).asList()
        lowerHalfFoods = []
        higherHalfFoods = []
        for food in foodList:
            if food[1] >= self.middleBorder:
                higherHalfFoods.append(food)
            else:
                lowerHalfFoods.append(food)
        return lowerHalfFoods, higherHalfFoods

    def ghostNextToMe(self, gameState):
        position = getPosition(gameState, self.index)
        neighbors = getNeighbors(gameState, position)

        enemyGhosts = self.observedEnemyGhosts(gameState)
        for neighbor in neighbors:
            for enemy in enemyGhosts:
                if getPosition(gameState, enemy) == neighbor:
                    return True
        return False

    def isBeingChasing(self, gameState):
        my_last_pos = getPosition(self.lastState, self.index)
        # mate_last_pos = getPosition(self.lastState, self.getTeamMateID(gameState))
        my_pos = getPosition(gameState, self.index)
        # mate_pos = getPosition(gameState, self.getTeamMateID(gameState))
        observedGhost_last = self.observedEnemyGhosts(self.lastState)
        observedGhost_current = self.observedEnemyGhosts(gameState)
        for ghost_i in observedGhost_last:
            for ghost_j in observedGhost_current:
                if ghost_i == ghost_j:
                    i_pos = getPosition(self.lastState, ghost_i)
                    j_pos = getPosition(gameState, ghost_j)
                    if self.getMazeDistance(my_pos, j_pos) <= self.getMazeDistance(my_last_pos, i_pos) <= 4:
                        return True
                    # if self.getMazeDistance(mate_pos, j_pos) <= self.getMazeDistance(mate_last_pos, i_pos) <= 4:
                    #     return True
        return False
        # else:
        #     return False


    def isRepeating(self, gameState):
        if len(self.statesQueue.list) < 10:
            return False
        allPositions = []
        for state in self.statesQueue.list:
            allPositions.append(getPosition(state, self.index))
        if allPositions[0] == allPositions[2] == allPositions[4] == allPositions[6] == allPositions[8] and \
                allPositions[1] == allPositions[3] == allPositions[5] == allPositions[7] == allPositions[9] and \
                isPacman(gameState, self.index):
            self.statesQueue.list = []
            return True
        else:
            return False

    def atBorder(self, gameState):
        borderList = self.homeBorderPosList
        pos = getPosition(gameState, self.index)
        if pos not in borderList:
            return False
        else:
            return True

    def getTeamMateID(self, gameState):
        myTeam = self.getTeam(gameState)
        if self.index == myTeam[0]:
            team_mate = myTeam[1]
        else:
            team_mate = myTeam[0]
        return team_mate

    def shortestPathToHome(self, gameState):
        all_positions_avoid = self.dangerousPositions(gameState)
        current_pos = getPosition(gameState, self.index)
        borders = self.homeBorderPosList
        shortest_path = None
        shortest_length = 10000
        for border in borders:
            path = self.aStarSearch(gameState, current_pos, [border], all_positions_avoid)
            if path is not None and len(path) < shortest_length:
                shortest_length = len(path)
                shortest_path = path
        if shortest_path is None:
            return []
        return shortest_path

    def shortestPathToHomeWhenStruggle(self, gameState):
        all_positions_avoid = self.mostDangerousPositions(gameState)
        current_pos = getPosition(gameState, self.index)
        borders = self.homeBorderPosList
        shortest_path = None
        shortest_length = 10000
        for border in borders:
            path = self.aStarSearch(gameState, current_pos, [border], all_positions_avoid)
            if path is not None and len(path) < shortest_length:
                shortest_length = len(path)
                shortest_path = path
        return shortest_path

    def inOurLand(self, pos):
        border_x = self.homeBorderPosList[0][0]
        if self.red:
            if pos[0] <= border_x:
                return True
            else:
                return False
        else:
            if pos[0] >= border_x:
                return True
            else:
                return False

##################################################################################
# Helper Functions Global #
##################################################################################

def getPosition(gameState, agentIndex):
    return gameState.getAgentState(agentIndex).getPosition()


def isPacman(gameState, agentIndex):
    return gameState.getAgentState(agentIndex).isPacman


def getScaredTime(gameState, agentIndex):
    return gameState.getAgentState(agentIndex).scaredTimer


def getHomeBorderPos(gameState, red):
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    halfway = width // 2
    x = halfway
    if red:
        x = halfway - 1

    # Walls
    borderPosList = []
    for y in range(1, height):
        if not gameState.hasWall(x, y):
            borderPosList.append((x, y))
    return borderPosList


def getEnemyBorderPos(gameState, red):
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    halfway = width // 2
    x = halfway
    if not red:
        x = halfway - 1

    # Walls
    borderPosList = []
    for y in range(1, height):
        if not gameState.hasWall(x, y):
            borderPosList.append((x, y))
    return borderPosList
def getMiddleBorder(gameState):
    height = gameState.data.layout.height
    return height / 2


def getNeighbors(gameState, pos):
    x, y = int(pos[0]), int(pos[1])
    left = (x - 1, y)
    right = (x + 1, y)
    up = (x, y + 1)
    down = (x, y - 1)
    neighborsList = [left, right, up, down]
    len_neighborList = len(neighborsList)
    notWallNeighborsSet = set(neighborsList)
    for i in range(len_neighborList):
        neighbor_x, neighbor_y = neighborsList[i]
        if gameState.hasWall(neighbor_x, neighbor_y):
            notWallNeighborsSet.remove((neighbor_x, neighbor_y))
    return notWallNeighborsSet


def printExecutionTime(start):
    # print("step time: ", time.time() - start)
    if time.time() - start > 0.2:
        print("time limit warning")
        print("step time: ", time.time() - start)

    # if time.time() - start > 1:
    #     print("time limit error")


"""
  def filterCornerPosSet(gameState, xrange, yrange)
    Get the coner position set corridor position set for given gameState and x,y
    range.

    Arguments:
    =========
        gameState: Current gameState
        xrange   : The range of x coordinates
        yrange   : The range of y coordinates

    Returns:
    ========
        cornerPosSet  : The position set of corner positions
        corridorPosSet: The position set of corridor positions
"""


def filterCornerPosSet(gameState, xrange, yrange):
    availableList = []
    neighborMap = {}
    cornerPosSet = set()
    for x in xrange:
        for y in yrange:
            if not gameState.hasWall(x, y):
                position = (x, y)
                neighborsSet = getNeighbors(gameState, position)
                neighborMap[position] = neighborsSet
                # if the num of neighbors is 1, this is a corner
                if 1 == len(neighborsSet):
                    cornerPosSet.add(position)
                else:
                    availableList.append(position)
    corridorPosSet = getCorridorPosSet(availableList,
                                       neighborMap,
                                       cornerPosSet)
    return cornerPosSet, corridorPosSet


"""
  def getCorridorPosSet(availableList, neighborMap, cornerPosSet):
    Get the corridor position set for given available list and coner position
    set.

    Arguments:
    =========
        availableList: Current available position list (not wall position)
        neighborMap  : The neighbor set map for each position in availableList.
        cornerPosSet : The corner position set based on availableList

    Returns:
    ========
        corridorPosSet: The position set of corridor positions
"""


def getCorridorPosSet(availableList, neighborMap, cornerPosSet):
    # if this position is next to the corner or corridor, and only two neighbors,
    # this is a corridor point
    corridorPosSet = set()
    lastAvailableNum = len(availableList)
    newAvailableNum = 0
    # when the num of last available positions are not change, exit the loop
    while lastAvailableNum != newAvailableNum:
        lastAvailableNum = len(availableList)
        availableListCopy = availableList.copy()
        for availablePos in availableListCopy:
            neighborsSet = neighborMap[availablePos]
            if len(neighborsSet) >= 2:
                deadNeighborCounter = 0
                for neighbor in neighborsSet:
                    if (neighbor in cornerPosSet) or (neighbor in corridorPosSet):
                        deadNeighborCounter += 1
                if deadNeighborCounter >= (len(neighborsSet) - 1):
                    availableList.remove(availablePos)
                    corridorPosSet.add(availablePos)
        newAvailableNum = len(availableList)
    return corridorPosSet


"""
    def getCornerAndCorridorDict(gameState, red)

    Arguments:
    =========
        gameState: current gameState
        red: A boolean value indicates whether is on red Team

    Returns:
    ========
        returnDict:
            returnDict['home']['cornerSet']:
                    The coner position set on home side
            returnDict['home']['corridorSet']
                    The corridor position set on home side
            returnDict['opponents']['cornerSet']
                    The coner position set on opponents side
            returnDict['opponents']['corridorSet']
                    The corridor position set on opponents side

    Corner:
        ---
        |0
        ---

    Corridor:
        ------
        |  0 0
        ------
"""


def getCornerAndCorridorDict(gameState, red):
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    halfway = width // 2
    yrange = range(1, height)
    redXrange = range(1, halfway)
    redCornerPosSet, redCorridorPosSet = \
        filterCornerPosSet(gameState, redXrange, yrange)

    blueXrange = range(halfway, width)
    blueCornerPosSet, blueCorridorPosSet = \
        filterCornerPosSet(gameState, blueXrange, yrange)

    returnDict = {'home': {}, 'opponents': {}}
    if red:
        returnDict['home']['cornerSet'] = redCornerPosSet
        returnDict['home']['corridorSet'] = redCorridorPosSet
        returnDict['opponents']['cornerSet'] = blueCornerPosSet
        returnDict['opponents']['corridorSet'] = blueCorridorPosSet
    else:
        returnDict['home']['cornerSet'] = blueCornerPosSet
        returnDict['home']['corridorSet'] = blueCorridorPosSet
        returnDict['opponents']['cornerSet'] = redCornerPosSet
        returnDict['opponents']['corridorSet'] = redCorridorPosSet
    return returnDict


##################################################################################
# Get Corner Gates Class #
##################################################################################


class Corner():
    def __init__(self, gameState, isRed):
        self.gameState = gameState
        self.isRed = isRed
        self.homeGateMap = None
        self.opponentGateMap = None
        self.initCorner()

    def getGatePosition(self, pos):
        gateMap = self.homeGateMap.copy()
        gateMap.update(self.opponentGateMap)
        for key, posSet in gateMap.items():
            if pos in posSet:
                return key
        return None

    def getHomeGateMap(self):
        return self.homeGateMap

    def getOpponentGateMap(self):
        return self.opponentGateMap

    def initCorner(self):
        width = self.gameState.data.layout.width
        height = self.gameState.data.layout.height
        halfway = width // 2
        yrange = range(1, height)
        redXrange = range(1, halfway)
        redGateMap = self.getCornerPosSet(redXrange, yrange)

        blueXrange = range(halfway, width)
        blueGateMap = self.getCornerPosSet(blueXrange, yrange)
        if self.isRed:
            self.homeGateMap = redGateMap
            self.opponentGateMap = blueGateMap
        else:
            self.homeGateMap = blueGateMap
            self.opponentGateMap = redGateMap

    def getCornerPosSet(self, xrange, yrange):
        neighborMap = {}
        cornerPosSet = set()
        for x in xrange:
            for y in yrange:
                if not self.gameState.hasWall(x, y):
                    position = (x, y)
                    neighborsSet = getNeighbors(self.gameState, position)
                    neighborMap[position] = neighborsSet
                    # if the num of neighbors is 1, this is a corner
                    if 1 == len(neighborsSet):
                        cornerPosSet.add(position)
        gateMap = self.expandCornerPosSet(cornerPosSet, neighborMap)
        return gateMap


    def expandCornerPosSet(self, cornerPosSet, neighborMap):
        cornerNeighborMap = {}
        closedSet = cornerPosSet.copy()
        candidateQueue = list()

        # For each corner, find its neighbors which are in dangerous position
        for corner in cornerPosSet:
          neighborsSet = neighborMap[corner]
          queue = list(neighborsSet)
          cornerNeighborMap[corner] = [corner]

          while 0 != len(queue):
            pos = queue.pop()
            x, y = pos
            if pos not in neighborMap.keys():
              neighborMap[pos] = getNeighbors(self.gameState, pos)
            tempNeighborsSet = neighborMap[pos]

            if 2 != len(tempNeighborsSet):
              candidateQueue.append(pos)
              continue
            else:
              closedSet.add(pos)
              cornerNeighborMap[corner].append(pos)

            for neighbor in tempNeighborsSet:
              if neighbor not in closedSet:
                queue.append(neighbor)

        ## Merge conneted corner cluster
        gateMap = {}
        while 0 != len(candidateQueue):
          pos = candidateQueue.pop()
          x, y = pos
          if pos not in neighborMap.keys():
            neighborMap[pos] = getNeighbors(self.gameState, pos)
          neighborsSet = neighborMap[pos]
          cornerCandidateSet = set()
          for neighbor in neighborsSet:
            for corner, cornerNeighborSet in cornerNeighborMap.items():
              if neighbor in cornerNeighborSet:
                cornerCandidateSet.add(corner)
                break
          # Merge corners
          tempCorner = cornerCandidateSet.pop()
          otherCorners = cornerCandidateSet
          closedSet.add(pos)
          for corner in otherCorners:
            cornerNeighborMap[tempCorner] += cornerNeighborMap.pop(corner)

          deadNeighborCounter = 0
          for neighbor in neighborsSet:
            if neighbor in cornerNeighborMap[tempCorner]:
              deadNeighborCounter += 1
          if (len(neighborsSet) - len(cornerCandidateSet)) > 2:
            if deadNeighborCounter >= (len(neighborsSet) - 1):
              cornerNeighborMap[tempCorner].append(pos)
              for neighbor in neighborsSet:
                if neighbor not in closedSet:
                  candidateQueue.append(neighbor)
            else:
            # Gate position
              gateMap[pos] = cornerNeighborMap[tempCorner]
          else:
          # Add new candidate to queue
            cornerNeighborMap[tempCorner].append(pos)
            for neighbor in neighborsSet:
              if neighbor not in closedSet:
                candidateQueue.append(neighbor)
        return gateMap





class HmmAgent():
  def __init__(self, gameState, opponentsIndeces):
    self.__beliefs = util.Counter()
    self.__legalPositions = None
    self.opponentsIndeces = opponentsIndeces
    self.numOpponents = len(opponentsIndeces)
    self.initializeBeliefs(gameState, opponentsIndeces)

  def initializeBeliefs(self, gameState, opponentsIndeces):
    for opponentIndex in opponentsIndeces:
      self.__beliefs[opponentIndex] = util.Counter()
      for p in self.getLegalPositions(gameState):
        self.__beliefs[opponentIndex][p] = 0.0
      initPos = gameState.getInitialAgentPosition(opponentIndex)
      self.__beliefs[opponentIndex][initPos] = 1.0
      self.__beliefs[opponentIndex].normalize()

  def getLegalPositions(self, gameState):
    if None == self.__legalPositions:
      self.__legalPositions = []
      walls = gameState.getWalls()
      for x in range(walls.width):
        for y in range(walls.height):
          if not walls[x][y]:
            self.__legalPositions.append((x, y))
    return self.__legalPositions

  def updateBelief(self, opponentIndex, belief):
    self.__beliefs[opponentIndex] = belief

  def getBelief(self, opponentIndex, position):
    return self.__beliefs[opponentIndex][position]

  def getBeliefs(self, opponentIndex):
    return self.__beliefs[opponentIndex]

  def getApproximateOpponentPos(self, opponentIndex):
    values = np.array(list(self.__beliefs[opponentIndex].values()))
    maxValue = max(values)
    keysArray = np.array(list(self.__beliefs[opponentIndex].keys()))
    maxKeyList = keysArray[values == maxValue]
    #maxAxisX = round(np.mean(maxKeyList[:,0]))
    #maxAxisY = round(np.mean(maxKeyList[:,1]))
    #return (maxAxisX, maxAxisY)
    return tuple(random.choice(maxKeyList))

  def getDifferentList(self, listA, listB):
    if listA == listB:
        return []
    if (len(listA) > len(listB)):
        return list(set(listA) - set(listB))
    else:
        return list(set(listB) - set(listA))

  def updateObservationList(self, agent, gameState):
    ## Update food defend position
    ## if Food is been eaten, the eaten food position is the ghost position
    newFoodDefend = agent.getFoodAndCapsuleDefend(gameState)
    diffFoodDefend = self.getDifferentList(newFoodDefend, agent.foodDefend)
    if 0 != len(diffFoodDefend):
      for foodPos in diffFoodDefend:
        candidateProb = np.zeros((self.numOpponents, 1))
        for i, opponentIndex in enumerate(self.opponentsIndeces):
          candidateProb[i] = self.getBelief(opponentIndex, foodPos)
        maxIndex = np.argmax(candidateProb)
        opponentIndex = self.opponentsIndeces[maxIndex]
        newBelief = util.Counter()
        newBelief[foodPos] = 0.99
        self.updateBelief(opponentIndex, newBelief)
      return

    ## Traditional update
    for opponentIndex in self.opponentsIndeces:
      myPosition = gameState.getAgentPosition(agent.index)
      newBelief = util.Counter()
      ghostPosition = gameState.getAgentPosition(opponentIndex)
      if ghostPosition != None:
        newBelief[ghostPosition] = 1
        self.updateBelief(opponentIndex, newBelief)
        continue

      noisyDistance = gameState.getAgentDistances()[opponentIndex]
      lastDistance = agent.getPreviousObservation()
      if lastDistance != None:
        lastDistance = lastDistance.getAgentDistances()[opponentIndex]
        if (lastDistance <= 3) and (noisyDistance - lastDistance > 10) and \
           (myPosition != gameState.getInitialAgentPosition(agent.index)) :
          self.initializeBeliefs(gameState, [opponentIndex])
          continue

      for pos in self.getLegalPositions(gameState):
        trueDistance = util.manhattanDistance(pos, myPosition)
        #distanceProb = gameState.getDistanceProb(trueDistance, noisyDistance)
        distanceProb = getObservationProbability(noisyDistance, trueDistance)
        if trueDistance <= 5:
          newBelief[pos] = 0
        elif 0 < distanceProb:
          #### HMM
          oldProbability = self.getBelief(opponentIndex, pos)
          x_coord, y_coord = pos
          possible_new_positions = [(x_coord+1, y_coord), (x_coord-1, y_coord),
                  (x_coord, y_coord+1), (x_coord, y_coord-1)]
          legal_new_positions = [position for position in possible_new_positions
                  if position in self.getLegalPositions(gameState)]
          prob = 1.0/len(legal_new_positions)
          for position in legal_new_positions:
              newBelief[pos] += (oldProbability + MIN_PROB)*prob

        else:
          newBelief[pos] = 0
      newBelief.normalize()
      self.updateBelief(opponentIndex, newBelief)


SONAR_NOISE_RANGE = 13 # Must be odd
SONAR_MAX = (SONAR_NOISE_RANGE - 1)/2
SONAR_NOISE_VALUES = [i - SONAR_MAX for i in range(SONAR_NOISE_RANGE)]
SONAR_DENOMINATOR = 2 ** SONAR_MAX  + 2 ** (SONAR_MAX + 1) - 2.0
SONAR_NOISE_PROBS = [2 ** (SONAR_MAX-abs(v)) / SONAR_DENOMINATOR  for v in SONAR_NOISE_VALUES]
MIN_PROB = 0.0001

observationDistributions = {}
def getObservationProbability(noisyDistance, trueDistance):
    """
    Returns the probability P( noisyDistance | trueDistance ).
    """
    global observationDistributions
    if noisyDistance not in observationDistributions:
        distribution = util.Counter()
        for error , prob in zip(SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
            distribution[max(1, noisyDistance - error)] += prob
        observationDistributions[noisyDistance] = distribution
    return observationDistributions[noisyDistance][trueDistance]

