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

from CheckersRL import random_ai


class solver:

    def __init__(self, step_size, epsilon, env, capacity, q_func, lr, gamma, player1,player2, env2):

        self.step_size = step_size
        self.epsilon = epsilon
        #self.env = checkers_env.checkers_env()
        self.env = env
        self.env2 = env2
        self.q_table = q_func.q_table
        #self.q_table = np.zeros(len(env.state_space), len(env.action_space))
        self.gamma = gamma             #0.99
        self.lr = lr                     #0.1
        self.epsilon = epsilon
        self.q_func = q_func
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.player = player1
        self.player2 = player2
    def solve(self, episode):
        total_reward = 0
        for i in episode:
            state = checkers_env.checkers_env.initialize_board()
            a = self.policy_improvement(state,self.epsilon,self.player)
            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward = self.env.step(a)
            # update Q value
            self.q_table[state, a] = (1-self.lr) * self.q_table[state, a] + self.lr*(reward + self.gamma*max(self.q_table[next_state,:]))
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
    def clear_memory(self):
        del self.memory
        self.memory = deque(maxlen=self.capacity)
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
                #print(type(max_action))
                #print(max_action)
                max_value = self.q_func.predict(state, max_action)
                for a in self.env.possible_actions(player):
                    #print("a:",self.q_func.predict(state, a))
                    #print("current_Q: ", max_value)
                    if self.q_func.predict(state, a) > max_value:
                        max_action = a
                        max_value = self.q_func.predict(state, a)
            else:
                return None
        else:
            try:
             max_action = random.sample(self.env.possible_actions(player), 1)[0]
            except Exception as e:
                print("error:")
                print(self.env.possible_actions(player), 1)
                #print(random.sample(self.env.possible_actions(player), 1)[0])


        return max_action

    def policy_evaluation(self,state,action):
        next_state, reward = self.env.step(action,self.player)
        next_state_index,next_action_index = self.q_func.tuple_to_index(next_state,action)
        #print("state-index: ",next_state_index,"state: ",next_state)
        #print("action-index: ", next_action_index,"action: ",action)
        #Q-Value for the action using bellman reward + gamma(max of q function) it rtrives all possible actions in the next state
        #print("len: ",len(self.q_func.q_table[0]))
        target = reward + self.gamma * np.max(self.q_func.q_table[next_state_index, next_action_index])
        td_error = target - self.q_func.predict(state, action)
        return state, action, next_state, target, td_error,reward

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = np.array(self.memory, dtype=object)
        print("batch:",type(batch))
        states = np.vstack(batch[:, 0])
        actions = np.array(batch[:, 1], dtype=int)
        next_states = np.vstack(batch[:, 2])
        targets = np.array(batch[:, 3], dtype=float)

        predicted_values = self.q_func.q_table[states, actions]
        self.q_func.q_table[states, actions] += self.step_size * (targets - predicted_values)

    def get_optimal_policy(self):
        optimal_policy = []
        for state_index in range(self.q_func.q_table.shape[0]):
            optimal_action = np.argmax(self.q_func.q_table[state_index, :])
            optimal_policy.append(optimal_action)
        #optimal_policy = np.argmax(self.q_func.q_table, axis=1)
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
    def solve2(self, episodes, epsilon, batch_size,numberOfGames):
      x = 0
      for epi in range(episodes):
        total_reward_per_episode = 0
       # x = 0
        for game in range(numberOfGames):
            state = self.env.initialize_board()
            player_2 = -1
            env2 = checkers_env.checkers_env(state, player_2)
            #x= 0
            total_reward = 0
            while not self.env.is_terminal(state):
                action = self.policy_improvement(state, epsilon,self.player)
                if action is None:
                    if self.env.game_winner(state) == self.player:
                        reward = 12
                    else:
                        reward = -12
                    total_reward += reward
                    print("len after 1 game", len(self.q_func.state_index_mapping))
                    print("actions taken: ", x)
                    break
                starters2 = env2.possible_pieces(self.player2)
                actions_player2 = env2.possible_actions(player_2)
                if len(actions_player2) == 0:
                   if env2.game_winner(state) == player_2:
                      total_reward += -12
                   else:
                    total_reward += 12
                    print("len after 1 game", len(self.q_func.state_index_mapping))
                    print("actions taken: ", x)
                    break
                else:
                 state, negate_reward = random_ai.make_a_move(possible_actions=actions_player2, env=env2, player=player_2)
                 #print("AI Move:")
                 #print(state)
                state, action, next_state, target, td_error,reward = self.policy_evaluation(state, action)
                #print("reward: ",reward)
                #print("RL Move:")
                #print(state)
                self.q_func.fit(state, action, self.lr, td_error)
                transition = self.generate_transition(state, action)
                self.push(transition)
                #print("action:",action)
                #self.experience_replay(batch_size)

                total_reward += reward
                total_reward_per_episode += total_reward

                state = next_state
                x+=1
        print(f"possiblestates: {x}")
            #print("len after 1 game",len(self.q_func.state_index_mapping))
            #print("actions taken: ", x)
            #self.experience_replay(batch_size)
        print(f"Episode {epi + 1}, Total Reward: {total_reward_per_episode}")






