import CNN
import checkers_env
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import checkers_env
import numpy as np

class solver:

    def __init__(self, step_size, epsilon, env, capacity, q_func,lr,gamma,player):

        self.step_size = step_size
        self.epsilon = epsilon
        #self.env = checkers_env.checkers_env()
        self.env = env
        self.q_table = q_func.q_table
        #self.q_table = np.zeros(len(env.state_space), len(env.action_space))
        self.gamma = gamma             #0.99
        self.lr = lr                     #0.1
        self.epsilon = epsilon
        self.q_func = q_func
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.player = player
    def solve(self, episode):
        total_reward = 0
        for i in episode:
            state = checkers_env.checkers_env.initialize_board()
            a = self.policy_improvement(state,self.epsilon,self.player)
            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward = self.env.step(a)
            # update Q value
            self.q_table[state, a] = (1-self.lr) * self.q_table[state, a] +self.lr*(reward + self.gamma*max(self.q_table[next_state,:]))
            total_reward += reward
            state = next_state
    #def policy_improvement(self):

    #   return a

    # def __init__(self, capacity, q_func, env, player):
    #     """
    #
    #     :param capacity: initialize replay memory to capacity
    #     :param q_func: initialize q_function, with input [state, action]
    #     """
    #     self.memory = deque([], maxlen=capacity)
    #     self.q_func = q_func
    #     self.env = checkers_env.checkers_env(env.initialize_board(), player)
    #     self.player = player

    def push(self, transition):
        """
        save a transition
        :param transition:
        :return:
        """
        self.memory.append(transition)

    def generate_transition(self, state, action):
        """
        execute action and observe reward and next state
        :param state:
        :param action:
        :return:
        """
        next_state,reward = self.env.step(action, self.player)
        transition = [state, action, next_state, reward]
        return transition


    def policy_improvement(self, state, epsilon, player):
        """
        With probability epsilon, choose a random action,
        otherwise, choose the action that maximizes Q.
        :param state:
        :param epsilon:
        :return: action
        """
        sample = random.random()
        if sample > epsilon:
            if len(self.env.possible_actions(player)) > 0:
                max_action = self.env.possible_actions(player)[0]
                print(type(max_action))
                print(max_action)
                max_value = self.q_func.predict(state, max_action)
                for a in self.env.possible_actions(player):
                    if self.q_func.predict(state, a) > max_value:
                        max_action = a
                        max_value = self.q_func.predict(state, a)
            else:
                return None
        else:
            max_action = random.sample(self.env.possible_actions(player), 1)[0]
        return max_action

    def policy_evaluation(self,state,action):
        next_state, reward = self.env.step(action,self.player)
        next_state_index = self.q_func.get_state_index(next_state)
        #Q-Value for the action using bellman reward + gamma(max of q function) it rtrives all possible actions in the next state
        target = reward + self.gamma * np.max(self.q_func.q_table[next_state_index, :])
        td_error = target - self.q_func.predict(state, action)
        return state, action, next_state, target, td_error,reward

    def experience_replay(self, batch_size):
        self.memory = []
        if len(self.memory) < batch_size:
            return

        batch = np.array(self.memory, dtype=object)
        states = np.vstack(batch[:, 0])
        actions = np.array(batch[:, 1], dtype=int)
        next_states = np.vstack(batch[:, 2])
        targets = np.array(batch[:, 3], dtype=float)

        predicted_values = self.q_func.q_table[states, actions]
        self.q_func.q_table[states, actions] += self.step_size * (targets - predicted_values)

    def get_optimal_policy(self):
        optimal_policy = np.argmax(self.q_func.q_table, axis=1)
        return optimal_policy

    def solve(self, episode, epsilon, batch_size):
        for t in episode:
            state = self.env.initialize_board()
            action = self.policy_improvement(state, epsilon, self.player)
            transition = self.policy_evaluation(state, action)
            state = transition[2]
            self.push(transition)
            samples = random.sample(self.memory, batch_size)
            self.policy_evaluation()
            self.q_func.fit()
    def solve2(self, episodes, epsilon, batch_size):
        for episode in range(episodes):
            total_reward = 0
            state = self.env.initialize_board()

            while not self.env.is_terminal(state):
                action = self.policy_improvement(state, epsilon,self.player)
                state, action, next_state, target, td_error,reward = self.policy_evaluation(state, action)

                self.q_func.fit(state, action, self.lr, td_error)
                transition = self.generate_transition(state,action)
                self.push(transition)
                self.experience_replay(batch_size)

                total_reward += reward

                state = next_state
                print(state)
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")






