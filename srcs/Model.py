import json
import numpy as np


class Model:
    input_layer: np.ndarray
    output_layer: np.ndarray

    def __init__(self, shape: list[int]) -> None:
        self.weights: np.ndarray = np.random.rand(shape)
        self.bias: np.ndarray = np.random.rand(shape)
        self.activations: np.ndarray = np.zeros(shape)
        self.gradients: np.ndarray = np.zeros(shape)
        self.nias_gradient: np.ndarray = np.zeros(shape)
        self.output_layer: np.ndarray = np.zeros(2)
        self.input_layer: np.ndarray = np.zeros(0)  # TODO: Apres analyse des donnees choisir l'input
        self.learning_rate: float = 0

    def get_layer(self, indice: int) -> tuple[list[float], list[float]]:
        return self.weights[indice], self.bias[indice]

    def softmax(self):
        """Output activation function"""
        return np.exp(self.output_layer) / np.sum(
            np.exp(self.output_layer), axis=0)

    def sigmoid(self, z: np.ndarray):
        """Compression function"""
        z_normalized: np.ndarray = np.zeros_like(z)
        for idx in range(len(z)):
            z_normalized[idx] = 1 / 1 + np.exp(-z[idx])
        return z_normalized

    def partial_derivative_sigmoid(self, z: np.ndarray):
        """Compute and return the partial derivative of z sigmoid"""
        sigmoid: np.ndarray = self.sigmoid(z)
        return np.array([sig * (1 - sig) for sig in sigmoid])

    def compute_activation(self):
        """Compute activation for the input given"""
        for idx in range(1, len(self.activations)):
            prev_activations: list[float] = self.activations[idx-1]
            new_activations: list[float] = []
            for neuron_weigths, neuron_bias in self.get_layer(idx):
                z = sum(w * a for w, a in zip(
                    neuron_weigths, prev_activations)
                    ) + neuron_bias
                new_activations.append(self.sigmoid(z))
            self.activations[idx] = new_activations

    def cross_entropy(self, ground_truth: np.ndarray) -> float:
        """Loss function"""
        return -np.sum(ground_truth * np.log(self.softmax()))

    def gradient_descent(self) -> None:
        """Weights update after backpropagation"""
        for idx in range(len(self.weights)):
            self.weights[idx] -= self.learning_rate * self.gradients[idx]
            self.bias[idx] -= self.gradients[idx]

    # TODO: prendre en compte le softmax de fin -> plutot fonction de cout

    def backpropagation(self) -> None:
        """
        Recursive Function which travels along layers,
        from the end to the beginning.
        Set self.gradients
        """
        for idx in range(len(self.gradients) - 2, -1, -1):
            w: np.ndarray = self.weights[idx]
            propagated_error: float = w.T @ self.gradients[idx+1]
            z: np.ndarray = self.weights[idx] @ self.activations[idx] + \
                self.bias[idx]
            self.gradients[idx] = propagated_error * \
                self.partial_derivative_sigmoid(z)
        self.gradient_descent()

    def train_step(self):
        pass

    def validation_step(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        """Save weights and bias into json file"""
        with open("model.json", "w+") as file:
            json_file: dict = {
                "weights": self.weights,
                "bias": self.bias
            }
            file.write(json.dumps(json_file))
