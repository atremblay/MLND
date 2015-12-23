import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict


class QState(object):
    """
    Q table for a specific state. It contains the value of every action
    for a state. This is a standalone object to manage the values of one
    single state.
    """
    def __init__(self, state):
        super(QState, self).__init__()
        self.state = state
        self.action = {action:0 for action in Environment.valid_actions}

    def __iter__(self):
        # Iterating over Environment.valid_actions just to keep order
        for action in Environment.valid_actions:
            yield self.action[action]

    def __getitem__(self, action):
        return self.action[action]

    def __setitem__(self, key, value):
        self.action[key] = value


class State(object):
    """docstring for State"""
    def __init__(self, next_waypoint, light, oncoming, left, right):
        super(State, self).__init__()
        self.next_waypoint = next_waypoint
        self.light = light
        self.oncoming = oncoming
        self.left = left
        self.right = right

    def __repr__(self):
        string = "State: next_waypoint = {}, light = {}, oncoming = {}, left = {}, right = {}"
        string = string.format(
            self.next_waypoint,
            self.light,
            self.oncoming,
            self.left,
            self.right)
        return string


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Will need a policy state to action

        def action_factory():
            return random.choice(Environment.valid_actions)
        self.policy = defaultdict(action_factory)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = State(self.next_waypoint, **inputs)
        print state

        # TODO: Select action according to your policy
        action = self.policy[state]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
