import json
import numpy as np


class Model:
    input_layer: np.ndarray
    output_layer: np.ndarray

    def __init__(self, layers_shape: list[int]) -> None:
        self.weights: np.ndarray = [
            np.random.randn(
                layers_shape[layer], layers_shape[layer-1]
            ) for layer in range(1, layers_shape)
        ]
        # Layers_shape[1:] pour ignorer l'input
        self.bias: np.ndarray = [
            np.zeros(layer, 1) for layer in layers_shape[1:]
        ]
        self.activations: np.ndarray = [
            np.zeros(layer, 1) for layer in layers_shape[1:]
        ]
        self.z: np.ndarray = [
            np.zeros(layer, 1) for layer in layers_shape[1:]
        ]
        self.grads: np.ndarray = [np.zeros_like(W) for W in self.weights]
        self.grads_b: np.ndarray = [np.zeros_like(b) for b in self.bias]
        self.output_activation: np.ndarray = np.zeros((layers_shape[-1], 1))
        self.softmax_output: np.ndarray = np.zeros((layers_shape[-1], 1))
        self.input_layer: np.ndarray = np.zeros((layers_shape[1], 1))
        self.ground_truth: np.ndarray = np.zeros(layers_shape[-1], 1)
        self.learning_rate: float = 0.7

    def __str__(self) -> str:
        print(f"Weights: {self.weights.shape}\n{self.weights}\n")
        print(f"Bias: {self.bias.shape}\n{self.bias}")
        print(f"Activations: {self.activations.shape}\n{self.activations}")
        print(f"Z: {self.z.shape}\n{self.z}")
        print(f"Grads: {self.grads.shape}\n{self.grads}")
        print(f"Bias Grads: {self.grads_b.shape}\n{self.grads_b}")
        print(f"Output Activations: {self.output_activation.shape}\n{self.output_activation}")
        print(f"Softmax Output: {self.softmax_output.shape}\n{self.softmax_output}")
        print(f"Input Layer: {self.input_layer.shape}\n{self.input_layer}")
        print(f"Ground Truth: {self.ground_truth.shape}\n{self.ground_truth}")
        print(f"Index: {self.index.shape}\n{self.index}")
        print(f"Label: {self.label}\n{self.label}")
        print(f"Learning Rate: {self.learning_rate}")

    class SizeDoNotMatch(Exception):
        def __init__(self, val1: str, val2: str):
            msg = f"Shape mismatch: {val1} and {val2} size do not match."
            super().__init__(msg)

    def get_layer(self, indice: int) -> tuple[list[float], list[float]]:
        """Return first weights then bias"""
        return self.weights[indice], self.bias[indice]

    def softmax(self, output_layer):
        """Output activation function"""
        exps = np.exp(output_layer - np.max(output_layer))
        return exps / np.sum(exps)

    def sigmoid(self, z: np.ndarray):
        """Compression function"""
        z_normalized: np.ndarray = np.zeros_like(z)
        for idx in range(len(z)):
            z_normalized[idx] = 1 / (1 + np.exp(-z[idx]))
        return z_normalized

    def partial_derivative_sigmoid(self, z: np.ndarray):
        """Compute and return the partial derivative of z sigmoid"""
        sigmoid: np.ndarray = self.sigmoid(z)
        return np.array([sig * (1 - sig) for sig in sigmoid])

    def compute_activation_layer(
        self, w: np.ndarray, prev_a: np.ndarray, bias: np.ndarray
    ):
        if len(w) != len(prev_a[0]):
            raise self.SizeDoNotMatch("weights", "previous activations")
        if len(w) != len(bias):
            raise self.SizeDoNotMatch("weights", "bias")
        return w @ prev_a + bias

    def compute_activation(self):
        """Compute activation for the input given"""
        for idx in range(len(self.activations)):
            if idx == 0:
                prev_activations: np.ndarray = self.input_layer
            else:
                prev_activations: np.ndarray = self.activations[idx-1]
            W, b = self.get_layer(idx)
            self.z[idx]: np.ndarray = self.compute_activation_layer(
                W, prev_activations, b)
            self.activations[idx] = self.sigmoid(self.z[idx])
        W, b = self.get_layer(len(self.activations))
        prev_activations = self.activations[-1]
        self.output_activation = self.compute_activation_layer(
            W, prev_activations, b
        )
        self.softmax_output = self.softmax(self.output_activation)

    # TODO: weird function, a travailler
    def cross_entropy(self, ground_truth: np.ndarray) -> float:
        """Loss function"""
        return -np.sum(ground_truth * np.log(self.softmax()))

    def gradient_descent(self) -> None:
        """Weights update after backpropagation"""
        for idx in range(len(self.weights)):
            self.weights[idx] -= self.learning_rate * self.grads[idx]
            self.bias[idx] -= self.learning_rate * self.grads_b[idx]

    def backpropagation(self) -> None:
        """

        """
        for idx in range(len(self.grads) - 1, -1, -1):
            if idx == len(self.grads) - 1:
                upper_grad: np.ndarray = self.output_softmax - self.ground_truth
            else:
                upper_grad: np.ndarray = self.grads[idx + 1]
            w: np.ndarray = self.weights[idx]
            if len(w) != len(upper_grad):
                raise self.SizeDoNotMatch("weights", "upper grad")
            propagated_error: np.ndarray = w.T @ upper_grad
            self.grads_b[idx] = propagated_error * \
                self.partial_derivative_sigmoid(self.z[idx])
            if len(self.grads_b[idx]) != len(self.activations[idx - 1]):
                raise self.SizeDoNotMatch(
                    "bias gradient", "previous activation")
            self.grads[idx] = self.grads_b[idx] @ self.activations[idx - 1].T
        self.gradient_descent()

    def train_step(self):
        self.compute_activation()
        return

    def validation_step(self):
        pass

    def train(self, input: np.ndarray, label: np.ndarray):
        if len(input) != len(self.input_layer):
            raise self.SizeDoNotMatch("input", "input_layer")
        self.input_layer = input
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
