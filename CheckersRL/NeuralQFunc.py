import torch
from torch import optim, nn

from CheckersRL.CNN import CNN


class NeuralQFunction():
    def __init__(self, cnn_model, state_size, action_size, q_table_file=None):
        self.cnn_model = cnn_model
        self.state_size = state_size
        self.action_size = action_size
        # Other initialization code...

    def predict(self, state, action):
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Forward pass through the CNN
        q_value = self.cnn_model(state_tensor)
        return q_value.item()

    def fit(self, state, action, lr, td_error):
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Forward pass to get the current Q-value
        current_q_value = self.cnn_model(state_tensor).item()
        # Update Q-value
        updated_q_value = current_q_value + lr * td_error
        # Backpropagation to adjust the neural network parameters
        loss = torch.nn.functional.mse_loss(torch.tensor(updated_q_value), self.cnn_model(state_tensor))
        loss.backward()
        # Optimize the model
        with torch.no_grad():
            for param in self.cnn_model.parameters():
                param.data -= lr * param.grad
        self.cnn_model.zero_grad()