import json
import pickle

import numpy as np
from scipy.sparse import lil_matrix



class q_function():
        def __init__(self, state_size, action_size, q_table_file=None):
            self.state_size = state_size
            self.action_size = action_size
            self.state_index_mapping = {}
            self.action_index_mapping = {}
            if q_table_file:
                # If a Q-table file is provided, load the Q-table
                with open(q_table_file, 'r') as file:
                    q_table_data = json.load(file)
                    self.q_table = self.load_q_table(q_table_data)
            else:
                self.q_table = lil_matrix((state_size, action_size), dtype=float)


        def get_state_index(self, state):
            state_key = str(state)
            if state_key not in self.state_index_mapping:
                new_index = len(self.state_index_mapping)
                self.state_index_mapping[state_key] = new_index
                # print(f"New state: {state}")
            # else:
            #     existing_index = self.state_index_mapping[state_key]
            #     print(f"Existing state: {state}, Index: {existing_index}, Hash: {hash(state_key)}")
            return self.state_index_mapping[state_key]

        def get_state_index_with_pieces(self, env):
            array_of_arrays = []
            array_of_arrays.append(env.possible_pieces(env.player))
            array_of_arrays.append(env.possible_pieces(-env.player))
            return array_of_arrays

        def get_action_index(self, action):
            action_key = str(action)
            if action_key not in self.action_index_mapping:
                new_index = len(self.action_index_mapping)
                self.action_index_mapping[action_key] = new_index
            # print(self.action_index_mapping[action_key])
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
            # print(f"Current Q-value: {current_q_value}, Updated Q-value: {updated_q_value}")
            self.q_table[state_index, action_index] = updated_q_value

        def save_q_table_json(self, q_table_file):
            q_table_data = {
                'q_table': self.q_table.toarray().tolist(),
                'state_index_mapping': self.state_index_mapping,
                'action_index_mapping': self.action_index_mapping
            }
            # Save the Q-table to a JSON file
            with open(q_table_file, 'w') as file:
                json.dump(q_table_data, file)

        def update_q_table_size(self, new_num_rows, new_num_columns):
            # Create a new sparse matrix with the updated size
            new_q_table = lil_matrix((new_num_rows, new_num_columns), dtype=float)

            # Copy values from the old matrix to the new one
            common_rows = min(self.q_table.shape[0], new_q_table.shape[0])
            common_columns = min(self.q_table.shape[1], new_q_table.shape[1])
            new_q_table[:common_rows, :common_columns] = self.q_table[:common_rows, :common_columns]

            # Update the Q-table reference
            self.q_table = new_q_table

        def load_q_table(self, q_table_data):
            # Create a new sparse matrix with the correct size
            #num_rows = len(q_table_data)
            new_q_table = lil_matrix((self.state_size, self.action_size), dtype=float)
            print(type(q_table_data))
            # Copy values from the loaded data to the new matrix
            #q_table_values = q_table_data.data
            #print(q_table_values["q_table"])
            # Convert the Q-table values from the list back to a NumPy array
            #q_table_values_array = np.array(q_table_values_list)

            for row_index, row_data in enumerate(q_table_data["q_table"]):
                for col_index, value in enumerate(row_data):
                    new_q_table[row_index, col_index] = value
            self.state_index_mapping = q_table_data['state_index_mapping']
            #print(len(self.state_index_mapping))
            self.action_index_mapping = q_table_data['action_index_mapping']
            return new_q_table