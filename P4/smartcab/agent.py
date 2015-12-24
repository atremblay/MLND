import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import random
import pdb
import math
from collections import Counter

class QTable(object):
    """docstring for QTable"""
    def __init__(self):
        super(QTable, self).__init__()
        self.qstates = dict()

    def __getitem__(self, state):
        assert isinstance(state, State)
        for s in self.qstates.keys():
            if state == s:
                return self.qstates[s]

        qstate = QState(state)
        self.qstates[state] = qstate
        return self.qstates[state]


class QState(object):
    """
    Q state for a specific state. It contains the value of every action
    for a state. This is a standalone object to manage the values of one
    single state.
    """
    def __init__(self, state):
        super(QState, self).__init__()
        assert isinstance(state, State)
        self.state = state
        self.actions = {
            action: Action(action) for action in Environment.valid_actions
        }

    def random(self):
        return random.choice(self.actions.values())

    def __iter__(self):
        for action in self.actions:
            yield self.actions[action]

    def __getitem__(self, action):
        return self.actions[action]

    def __setitem__(self, action, value):
        print("setting {} to {}".format(action, value))
        self.actions[action].value = value

    def __repr__(self):
        # pdb.set_trace()
        # string = "QState: next_waypoint = {}, light = {}, oncoming = {}, left = {}, right = {}"
        # string = string.format(
        #     self.state.next_waypoint,
        #     self.state.light,
        #     self.state.oncoming,
        #     self.state.left,
        #     self.state.right)

        string = "Q{}".format(self.state)
        for action in self.actions:
            string += "\n\t" + repr(self.actions[action])
        return string

    def __eq__(self, other):

        if not isinstance(other, QState):
            return False

        equal = True
        equal &= self.state == other.state
        return equal


class State(object):
    """docstring for State"""
    def __init__(self, next_waypoint, light, oncoming, left, right, deadline):
        super(State, self).__init__()
        self.next_waypoint = next_waypoint
        self.light = light
        self.oncoming = oncoming
        self.left = left
        self.right = right
        self.deadline = deadline

    def __repr__(self):
        string = "State: next_waypoint = {}, light = {}, oncoming = {}, left = {}, right = {}, deadline = {}"
        string = string.format(
            self.next_waypoint,
            self.light,
            self.oncoming,
            self.left,
            self.right,
            self.deadline)
        return string

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        equal = True
        equal &= self.next_waypoint == other.next_waypoint
        equal &= self.light == other.light
        equal &= self.oncoming == other.oncoming
        equal &= self.left == other.left
        equal &= self.right == other.right
        equal &= self.deadline == other.deadline
        return equal


class Action(object):
    """docstring for Action"""
    def __init__(self, action):
        super(Action, self).__init__()
        self.action = action
        self.value = 0

    def __lt__(self, other):
        if self.value == other.value:
            return random.choice([True, False])
        return self.value < other.value

    def __repr__(self):
        return "action: {}, value: {}".format(self.action, self.value)


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma = 0.8

        # def action_factory():
        #     return random.choice(Environment.valid_actions)
        # self.policy = defaultdict(action_factory)
        self.qtable = QTable()
        self.ratio = list()
        self.count = Counter()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if self.count["GOOD"] == 0 and self.count["WRONG"] == 0:
            return
        self.ratio.append(float(self.count["WRONG"])/(self.count["GOOD"]+self.count["WRONG"]))
        print(self.ratio)
        self.count = Counter()

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        deadline = self.env.get_deadline(self)
        # pdb.set_trace()

        # TODO: Update state
        state = State(self.next_waypoint, deadline=deadline, **inputs)
        qstate = self.qtable[state]
        # print qstate

        # TODO: Select action according to your policy
        #  Boltzmann exploration

        T = .75
        num = math.exp(max(qstate).value/T)
        denom = sum([math.exp(action.value/T) for action in qstate])
        pr_a_s = num/denom
        print("======================================")
        print("Probability of being greedy: {}".format(pr_a_s))
        print(qstate)
        if random.random() > pr_a_s:
            action = qstate.random()
            print("Exploration")
            # self.count["Exploration"] += 1
        else:
            action = max(qstate)
            # self.count["Exploitation"] += 1
            print("Exploitation")


        # Execute action and get reward
        reward = self.env.act(self, action.action)
        if reward != -1:
            print("GOOD ACTION")
            self.count["GOOD"] += 1
        else:
            print("WRONG ACTION")
            self.count["WRONG"] += 1

        # TODO: Learn policy based on state, action, reward
        state_prime = State(
            self.planner.next_waypoint(),
            deadline=deadline,
            **self.env.sense(self)
            )
        qstate_prime = self.qtable[state_prime]
        self.state = state_prime

        qsa = action.value
        qsa_prime = max(qstate_prime).value  # Return the highest value of all actions

        #Q[s,a] <- Q[s,a] + \alpha*(R(s,a)+ \gamma*max_a' Q[s',a'] - Q[s,a])
        # Q[s,a]<-(1-alpha) Q[s,a] + alpha(r+ gamma*max_a' Q[s',a']).
        # self.qtable[state][action.action] = qsa + self.alpha * (reward + self.gamma*qsa_prime - qsa)
        alpha = 1./(t+1)
        self.qtable[state][action.action] = (1-alpha)*qsa + alpha*(reward + self.gamma*qsa_prime)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=1001)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
