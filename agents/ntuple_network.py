import numpy as np
import torch

class NTupleNetwork:
    def __init__(self, n_tuples, optimistic_value=1000):
        """
        n_tuples: a list of lists, each inner list holds the board indices (0..15) that form one tuple.
        optimistic_value: initial high value to encourage exploration.
        """
        self.n_tuples = n_tuples
        # For each tuple, use a dictionary to simulate a lookup table (LUT).
        self.weights = [dict() for _ in self.n_tuples]
        self.optimistic_value = optimistic_value
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device):
        """Move the network to the specified device."""
        self.device = device
        return self

    def evaluate(self, state: torch.Tensor) -> float:
        """Evaluate state value by summing over n-tuple features."""
        total_value = 0.0
        flat_state = state.flatten()
        for idx, tuple_indices in enumerate(self.n_tuples):
            # Extract the feature (a tuple of cell values)
            feature = tuple(flat_state[i].item() for i in tuple_indices)
            # If feature not seen before, use optimistic initialization.
            w = self.weights[idx].get(feature, self.optimistic_value)
            total_value += w
        return total_value

    def update_weights(self, state: torch.Tensor, td_error: float, learning_rate: float):
        """Distribute the TD error equally to the weights corresponding to each tuple feature."""
        flat_state = state.flatten()
        for idx, tuple_indices in enumerate(self.n_tuples):
            feature = tuple(flat_state[i].item() for i in tuple_indices)
            current_w = self.weights[idx].get(feature, self.optimistic_value)
            new_w = current_w + learning_rate * td_error / len(self.n_tuples)
            self.weights[idx][feature] = new_w

class OptimisticTDAgent:
    def __init__(self, n_tuples, learning_rate=0.001, optimistic_value=1000):
        self.network = NTupleNetwork(n_tuples, optimistic_value)
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, device):
        """Move the agent to the specified device."""
        self.device = device
        self.network.to(device)
        return self

    def evaluate(self, state: torch.Tensor) -> float:
        return self.network.evaluate(state)

    def update(self, state: torch.Tensor, reward: float, next_state: torch.Tensor, terminal: bool) -> float:
        """
        Perform a TD update:
          TD error = (reward + next_value) - current_value
        """
        current_value = self.evaluate(state)
        next_value = 0 if terminal else self.evaluate(next_state)
        td_error = (reward + next_value) - current_value
        self.network.update_weights(state, td_error, self.learning_rate)
        return td_error 