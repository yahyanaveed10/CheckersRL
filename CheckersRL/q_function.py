import numpy as np


class q_function():
    def __init__(self,state_size, action_size):
        self.q_table = np.zeros((state_size, action_size), dtype=float)
        self.state_index_mapping = {}
        self.action_index_mapping = {}

    def get_state_index(self, state):
        state_key = tuple(map(tuple, state))
        if state_key not in self.state_index_mapping:
            new_index = len(self.state_index_mapping)
            self.state_index_mapping[state_key] = new_index
        return self.state_index_mapping[state_key]

    def get_action_index(self, action):
        action_key = tuple(action)
        if action_key not in self.action_index_mapping:
            new_index = len(self.action_index_mapping)
            self.action_index_mapping[action_key] = new_index
        return self.action_index_mapping[action_key]

    def tuple_to_index(self, state, action):
        state_index = self.get_state_index(state)
        action_index = self.get_action_index(action)
        return state_index, action_index

    def predict(self, state, action):
        state_index, action_index = self.tuple_to_index(state, action)
        return self.q_table[state_index, action_index]

    def fit(self, state, action, lr, td_error):
        state_index, action_index = self.tuple_to_index(state, action)
        current_q_value = self.q_table[state_index, action_index]
        updated_q_value = current_q_value + lr * td_error

        self.q_table[state_index, action_index] = updated_q_value