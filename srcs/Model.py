import json
import numpy as np


class Model:
    input_layer: np.ndarray
    output_layer: np.ndarray

    def __init__(self, layers_shape: list[int], batch_size: int) -> None:
        # Layers_shape[1:] pour ignorer l'input
        print(f"LAYER SHAPE: {layers_shape}")
        for layer in range(1, len(layers_shape)):
            print(f" Layer: {layer}")
            print(f"Cur layer shape: {layers_shape[layer]}")
            print(f"Prev Layer Shape: {layers_shape[layer-1]}")
        self.weights: np.ndarray = [
            np.random.randn(
                layers_shape[layer], layers_shape[layer-1]
            ) for layer in range(1, len(layers_shape))
        ]
        self.output_weights: np.ndarray = np.random.randn(
            layers_shape[-1], layers_shape[-2])
        self.bias: np.ndarray = [
            np.zeros((layer, 1)) for layer in layers_shape[1:]
        ]
        self.bias_output = np.zeros((layers_shape[-1], 1))
        
        self.activations: np.ndarray = [
            np.zeros((layer, batch_size)) for layer in layers_shape[1:]
        ]
        self.z: np.ndarray = [
            np.zeros((layer, batch_size)) for layer in layers_shape[1:]
        ]
        self.grads: np.ndarray = [np.zeros_like(W) for W in self.weights]
        self.grads_b: np.ndarray = [np.zeros_like(b) for b in self.bias]
        self.output_activation: np.ndarray = np.zeros(
            (layers_shape[-1], batch_size))
        self.softmax_output: np.ndarray = np.zeros(
            (layers_shape[-1], batch_size))
        self.input_layer: np.ndarray = np.zeros(
            (layers_shape[0], batch_size))
        self.ground_truth: np.ndarray = np.zeros(
            (layers_shape[-1], batch_size))
        self.learning_rate: float = 0.7
        self.bacth_size: int = batch_size

    def __str__(self) -> str:
        msg = ""
        msg += f"Weights: ({len(self.weights)}, {len(self.weights[0])})\n"
        msg += f"Weights Last Layer: ({len(self.output_weights)}, "
        msg += f"{len(self.output_weights[0])})\n"
        msg += f"Bias: ({len(self.bias)}, {len(self.bias[0])})\n"
        msg += f"Bias Last Layer: ({len(self.bias_output)}, "
        msg += f"{len(self.bias_output[0])})\n"
        msg += f"Activations: ({len(self.activations)}, "
        msg += f"{len(self.activations[0])})\n"
        msg += f"Z: ({len(self.z)}, {len(self.z[0])})\n"
        msg += f"Grads: ({len(self.grads)}, {len(self.grads[0])})\n"
        msg += f"Bias Grads: ({len(self.grads_b)}, {len(self.grads_b[0])})\n"
        msg += "Output Activations: ("
        msg += f"{len(self.output_activation)}, "
        msg += f"{len(self.output_activation[0])})\n"
        msg += f"Softmax Output: ({len(self.softmax_output)}, "
        msg += f"{len(self.softmax_output[0])})\n"
        msg += f"Input Layer: ({len(self.input_layer)}, "
        msg += f"{len(self.input_layer[0])})\n"
        msg += f"Ground Truth: ({len(self.ground_truth)}, "
        msg += f"{len(self.ground_truth[0])})\n"
        msg += f"Learning Rate: {self.learning_rate}\n"
        msg += f"Batch_size: {self.bacth_size}"
        return msg
    
    def print_shape(self, array: np.ndarray):
        copy = np.zeros_like(array)
        print(copy)

    class SizeDoNotMatch(Exception):
        def __init__(self, val1: str, val2: str):
            msg = f"Shape mismatch: {val1} and {val2} size do not match."
            super().__init__(msg)

    def get_layer(self, indice: int) -> tuple[list[float], list[float]]:
        """Return first weights then bias"""
        print(f"INDICE: {indice}")
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
        if w.shape[1] != prev_a.shape[0]:
            print(f"W SHAPE {w.shape}")
            print(f"A-1 SHAPE: {prev_a.shape}")
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
            print(f"SHAPE ACT IDX {idx}: {self.activations[idx].shape}")
        # Last activations
        prev_activations = self.activations[-1]
        self.output_activation = self.compute_activation_layer(
            self.output_weights, prev_activations, self.bias_output
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
                upper_grad: np.ndarray = \
                    self.output_softmax - self.ground_truth
            else:
                upper_grad: np.ndarray = self.grads[idx + 1]
            w: np.ndarray = self.weights[idx]
            if len(w) != len(upper_grad):
                raise self.SizeDoNotMatch("weights", "upper grad")
            propagated_error: np.ndarray = w.T @ upper_grad
            tmp_grads_b = propagated_error * \
                self.partial_derivative_sigmoid(self.z[idx])
            if len(self.grads_b[idx]) != len(self.activations[idx - 1]):
                raise self.SizeDoNotMatch(
                    "bias gradient", "previous activation")
            # Division par batch_size pour avoir une moyenne des exemple du batch
            self.grads[idx] = (
                tmp_grads_b @ self.activations[idx - 1].T
            ) / self.bacth_size
            # Moyenne sur les lignes donc moyennes du batch
            # (les colonnes sont les exemples,
            # les lignes representes les neuronnes actuels)
            self.grads_b[idx] = np.mean(tmp_grads_b, axis=1, keepdims=True)
        self.gradient_descent()

    def train_step(self):
        self.compute_activation()
        return

    def validation_step(self):
        pass

    def train(self, input: np.ndarray, label: np.ndarray):
        if input.shape[0] != self.input_layer.shape[0]:
            raise self.SizeDoNotMatch("input", "input_layer")
        if label.shape != self.ground_truth.shape:
            raise self.SizeDoNotMatch("label", "ground_truth")
        print(self)
        print(self.weights)
        print(self.activations)
        print(self.output_weights)
        print(self.bias_output)
        # Pour le dernier batch ou cas ou il est incomplet
        self.input_layer[:, :input.shape[1]] = input
        self.ground_truth = label
        self.compute_activation()
        print(self)

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
