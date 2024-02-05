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
        if len(self.env.possible_actions(player)) == 0:
            return  None
        sample = random.random()
        if random.uniform(0, 1) > self.epsilon:
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


    def solve2(self, episodes, epsilon, batch_size,numberOfGames):
      x = 0
      games_lost_total = 0
      games_won_total = 0
      games_drawn_total = 0
      for epi in range(episodes):
        total_reward_per_episode = 0
       # x = 0
        game_won = 0
        games_lost = 0
        games_drawn = 0
        for game in range(numberOfGames):
            state = self.env.initialize_board()
            player_2 = -1
            env2 = checkers_env.checkers_env(state, player_2)
            #x= 0
            total_reward = 0
            reward_dummy = 0
            while not self.env.is_terminal(state):
                negate_reward = 0
                action = self.policy_improvement(state, epsilon,self.player)
                if action is None:
                   if reward_dummy == 0:
                         if self.env.game_winner(state) == self.player:
                           reward = 12
                           total_reward += reward
                         else:
                           reward = -12
                           total_reward += reward
                    #total_reward_per_episode += total_reward
                    #print("RL ran out of moves and ai had a move left")
                   break
                starters2 = env2.possible_pieces(self.player2)
                 #print("AI Move:")
                 #print(state)
                state, action, next_state, target, td_error,reward = self.policy_evaluation(state, action)
                #print("reward: ",reward)
                #print("RL Move:")
                #print(state)
                state = next_state
                self.q_func.fit(state, action, self.lr, td_error)
                #transition = self.generate_transition(state, action)
                #self.push(transition)
                actions_player2 = env2.possible_actions(player_2)
                if len(actions_player2) == 0:
                    if len(self.env.possible_actions(self.player)) > 0:
                       if reward == 0 or reward == 2:
                          reward += 12
                    total_reward += reward
                    #total_reward_per_episode += total_reward
                    #print("AI is out of moves Rl won")
                    #print("len after 1 game", len(self.q_func.state_index_mapping))
                    #print("actions taken: ", x)
                    break
                else:
                 state, negate_reward = random_ai.make_a_move(possible_actions=actions_player2, env=env2, player=player_2)

                if negate_reward == 2:
                    total_reward += -negate_reward
                #print("action:",action)
                #self.experience_replay(batch_size)
                #print("reward each game: ", reward-negate_reward)
                reward_dummy = reward
                total_reward += reward
                #total_reward_per_episode += total_reward

                #state = next_state
                x+=1

            if self.env.game_winner(state)== self.player:
                    game_won +=1
                    print("game won,state:  \n ",state)
            elif self.env.game_winner(state)== -self.player:
                    games_lost += 1
                    print("game lost,state: \n", state)
            elif self.env.game_winner(state) == 0:
                    games_drawn += 1
                    print("game drawn,state: \n", state)
                    #print(state,"\npossible actions:", env2.possible_actions(player_2))
                   # print("game won, reward", reward)
            total_reward_per_episode += total_reward
            print("reward each game: ",total_reward)
        #print(f"possiblestates: {x}")
            #print("len after 1 game",len(self.q_func.state_index_mapping))
            #print("actions taken: ", x)
            #self.experience_replay(batch_size)
        print(f"Episode {epi + 1}, Total Reward: {total_reward_per_episode}")
        print("games won per episode: ", game_won)
        print("games lost per episode", games_lost)
        print("games drawn per episode", games_drawn)
        #print("reward : ", self.env.count)
        self.env.count = 0
        games_won_total += game_won
        games_lost_total += games_lost
        games_drawn_total += games_drawn
        #print("games won: ", game_won )
        #print("games lost ", games_lost)
      print("games won: ", games_won_total )
      print("games lost ", games_lost_total)
      print("games drawn ", games_drawn_total)




