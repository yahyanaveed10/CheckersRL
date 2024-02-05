import random
from collections import deque

import numpy as np
import torch

from CheckersRL import random_ai, checkers_env
from CheckersRL.CNN import CNN


class NeuralQSolver:

    def __init__(self, step_size, epsilon, env, capacity, input_size,output_size, lr, gamma, player1, player2, env2):

        self.step_size = step_size
        self.epsilon = epsilon
        self.env = env
        self.env2 = env2
        self.q_func = CNN(input_size=input_size, output_size=output_size)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.player = player1
        self.player2 = player2

    def push(self, transition):
        """
        Save a transition
        :param transition:
        :return:
        """
        self.memory.append(transition)

    def generate_transition(self, state, action):
        """
        Execute action and observe reward and next state
        :param state:
        :param action:
        :return:
        """
        next_state, reward = self.env.step(action, self.player)
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
        if random.uniform(0, 1) > self.epsilon:
            if len(self.env.possible_actions(player)) > 0:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = self.q_func(state_tensor)
                max_action = torch.argmax(q_values).item()
            else:
                return None
        else:
            try:
                max_action = random.sample(self.env.possible_actions(player), 1)[0]
            except Exception as e:
                print("Error:", e)
                return None

        return max_action

    def policy_evaluation(self, state, action):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_func(state_tensor)
        predicted_value = q_values[action].item()

        next_state, reward = self.env.step(action, self.player)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        # Calculate the target Q-value using Bellman equation
        target = reward + self.gamma * torch.max(self.q_func(next_state_tensor)).item()
        td_error = target - predicted_value

        return state, action, next_state, target, td_error, reward

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = np.array(self.memory, dtype=object)
        states = np.vstack(batch[:, 0])
        actions = np.array(batch[:, 1], dtype=int)
        next_states = np.vstack(batch[:, 2])
        targets = np.array(batch[:, 3], dtype=float)

        for i in range(batch_size):
            state_tensor = torch.Tensor(states[i])
            action_tensor = torch.Tensor([actions[i]])
            input_tensor = torch.cat((state_tensor, action_tensor), dim=0)
            target_tensor = torch.Tensor([targets[i]])
            self.q_func.optimizer.zero_grad()
            output = self.q_func.cnn(input_tensor)
            loss = self.q_func.criterion(output, target_tensor)
            loss.backward()
            self.q_func.optimizer.step()

    def get_optimal_policy(self):
        optimal_policy = []
        for state_index in range(self.q_func.state_size):
            optimal_action = np.argmax(self.q_func.predict(state_index, np.arange(self.q_func.action_size)).detach().numpy())
            optimal_policy.append(optimal_action)
        return optimal_policy

    def solve(self, episodes, epsilon, batch_size, number_of_games):
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
            for game in range(number_of_games):
                state = self.env.initialize_board()
                player_2 = -1
                env2 = checkers_env.checkers_env(state, player_2)
                # x= 0
                total_reward = 0
                reward_dummy = 0
                while not self.env.is_terminal(state):
                    negate_reward = 0
                    action = self.policy_improvement(state, epsilon, self.player)
                    if action is None:
                        if reward_dummy == 0:
                            if self.env.game_winner(state) == self.player:
                                reward = 12
                                total_reward += reward
                            else:
                                reward = -12
                                total_reward += reward
                        # total_reward_per_episode += total_reward
                        # print("RL ran out of moves and ai had a move left")
                        break
                    starters2 = env2.possible_pieces(self.player2)
                    # print("AI Move:")
                    # print(state)
                    state, action, next_state, target, td_error, reward = self.policy_evaluation(state, action)
                    # print("reward: ",reward)
                    # print("RL Move:")
                    # print(state)
                    state = next_state
                    self.q_func.fit(state, action, self.lr, td_error)
                    # transition = self.generate_transition(state, action)
                    # self.push(transition)
                    actions_player2 = env2.possible_actions(player_2)
                    if len(actions_player2) == 0:
                        if len(self.env.possible_actions(self.player)) > 0:
                            if reward == 0 or reward == 2:
                                reward += 12
                        total_reward += reward
                        # total_reward_per_episode += total_reward
                        # print("AI is out of moves Rl won")
                        # print("len after 1 game", len(self.q_func.state_index_mapping))
                        # print("actions taken: ", x)
                        break
                    else:
                        state, negate_reward = random_ai.make_a_move(possible_actions=actions_player2, env=env2,
                                                                     player=player_2)

                    if negate_reward == 2:
                        total_reward += -negate_reward
                    # print("action:",action)
                    # self.experience_replay(batch_size)
                    # print("reward each game: ", reward-negate_reward)
                    reward_dummy = reward
                    total_reward += reward
                    # total_reward_per_episode += total_reward

                    # state = next_state
                    x += 1

                if self.env.game_winner(state) == self.player:
                    game_won += 1
                    print("game won,state:  \n ", state)
                elif self.env.game_winner(state) == -self.player:
                    games_lost += 1
                    print("game lost,state: \n", state)
                elif self.env.game_winner(state) == 0:
                    games_drawn += 1
                    print("game drawn,state: \n", state)
                    # print(state,"\npossible actions:", env2.possible_actions(player_2))
                # print("game won, reward", reward)
                total_reward_per_episode += total_reward
                print("reward each game: ", total_reward)
            # print(f"possiblestates: {x}")
            # print("len after 1 game",len(self.q_func.state_index_mapping))
            # print("actions taken: ", x)
            # self.experience_replay(batch_size)
            print(f"Episode {epi + 1}, Total Reward: {total_reward_per_episode}")
            print("games won per episode: ", game_won)
            print("games lost per episode", games_lost)
            print("games drawn per episode", games_drawn)
            # print("reward : ", self.env.count)
            self.env.count = 0
            games_won_total += game_won
            games_lost_total += games_lost
            games_drawn_total += games_drawn
            # print("games won: ", game_won )
            # print("games lost ", games_lost)
        print("games won: ", games_won_total)
        print("games lost ", games_lost_total)
        print("games drawn ", games_drawn_total)
