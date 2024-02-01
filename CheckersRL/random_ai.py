import solver
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import checkers_env
import q_function

import random

import matplotlib.pyplot as plt
import numpy as np
import CNN


def make_a_move(possible_actions, env, player):
    selected_action = random.choice(possible_actions)
    #print(selected_action)
    return env.step(selected_action, -1)
    