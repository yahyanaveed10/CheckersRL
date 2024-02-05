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
from CheckersRL import NeuralQFunc, NNSolver

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
#print(actions_player1)
env.step([1, 1, 2, 0], 1)
# env2.step(actions_player2 = env.possible_actions(player=-1))
env.render()
state_space_size = len(env.board.flatten()) * 1000000
action_space_size = len(env.possible_actions(1)) * 15
q_func = q_function.q_function(state_space_size, action_space_size,"final_q_table.json")
solver_var = solver.solver(step_size=0.1, epsilon=0.1, env=env, capacity=1000, q_func=q_func, lr=0.1, gamma=0.99,
                           player1=1, player2=player_2, env2=env2)
#n_positions = 36
#num_actions = 15
#board_size = 6 * 6
#output_size = num_actions * board_size
#appro = CNN.CNN(36,2)
#q_func_neural = NeuralQFunc.NeuralQFunction(state_space_size, output_size,None)
#nn_solver_var = NNSolver.NeuralQSolver(step_size=0.1, epsilon=0.1, env=env, capacity=1000, input_size=36,output_size=2, lr=0.1, gamma=0.99,
#                           player1=1, player2=player_2, env2=env2)
#nn_solver_var.solve(episodes=1, epsilon=0.1, batch_size=32,number_of_games=10)
# Train the solver
solver_var.solve2(episodes=1, epsilon=0.1, batch_size=32,numberOfGames=10) ###add a breaker to change the value of epsiolon to 0.1
new_num_rows = len(q_func.state_index_mapping)
new_num_columns = len(q_func.action_index_mapping)
q_func.update_q_table_size(new_num_rows, new_num_columns)
print("total number of states:", len(solver_var.q_func.state_index_mapping))
print("total number of actions:", len(solver_var.q_func.action_index_mapping))
print("total number of rows:", solver_var.q_func.q_table.shape[0])
# Get the optimal policy
#optimal_policy = solver_var.get_optimal_policy()
#print("Optimal Policy:", optimal_policy)
q_func.save_q_table_json('final_q_table.json')
print("total number of states:", len(solver_var.q_func.state_index_mapping))
solver_var.clear_memory()


# batch_size is the number of transitions sampled from the replay buffer
batch_size = 64
n_positions = 36

#appro = CNN.CNN(n_positions)
# appro.load_state_dict()
# appro.load_
#
# def logistic(samples, targets):
#     clf = LogisticRegression()
#     clf.fit(samples, targets)
#     return clf
