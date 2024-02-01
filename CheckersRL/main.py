import solver
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import checkers_env
import q_function
import random_ai
import random

import matplotlib.pyplot as plt
import numpy as np
import CNN

board = np.array([[0, 1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, -1, 0, -1, 0, -1],
                  [-1, 0, -1, 0, -1, 0]], dtype=int)
"""
def initialize_board():
    # 1 and -1 represent the pieces of two players 1 and -1
    board = np.zeros((6, 6))
    for i in range(2):
        for j in range(0, 6, 2):
            board[i][j + (i % 2)] = 1
            board[6 - i - 1][j + (i % 2)] = -1
    return board
"""
player_2 = -1
env2 = checkers_env.checkers_env(board, player_2)
starters2 = env2.possible_pieces(player_2)
actions_player2 = env2.possible_actions(player_2)
random_ai.make_a_move(possible_actions=actions_player2, env=env2, player=player_2)

env = checkers_env.checkers_env(board, 1)
# env.board = env.initialize_board()
env.render()
starters = env.possible_pieces(1)
actions_player1 = env.possible_actions(player=1)
print(actions_player1)
env.step([1, 1, 2, 0], 1)
# env2.step(actions_player2 = env.possible_actions(player=-1))
env.render()
state_space_size = len(env.board.flatten()) * 500
action_space_size = len(env.possible_actions(1)) * 500
q_func = q_function.q_function(state_space_size, action_space_size)
solver_var = solver.solver(step_size=0.1, epsilon=0.1, env=env, capacity=1000, q_func=q_func, lr=0.1, gamma=0.99,
                           player1=1, player2=player_2, env2=env2)

# Train the solver
solver_var.solve2(episodes=1000, epsilon=0.1, batch_size=32)

# Get the optimal policy
optimal_policy = solver_var.get_optimal_policy()
print("Optimal Policy:", optimal_policy)

# batch_size is the number of transitions sampled from the replay buffer
batch_size = 64
n_positions = 36

appro = CNN.CNN(n_positions)
# appro.load_state_dict()
# appro.load_
#
# def logistic(samples, targets):
#     clf = LogisticRegression()
#     clf.fit(samples, targets)
#     return clf
