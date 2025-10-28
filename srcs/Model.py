import json
import numpy as np


class Model:
    input_layer: np.ndarray
    output_layer: np.ndarray

    def __init__(self, layers_shape: list[int], batch_size: int, learning_rate: float) -> None:
        # Layers_shape[1:] pour ignorer l'input

        self.weights: np.ndarray = [
            np.random.randn(
                layers_shape[layer], layers_shape[layer-1]
            ) for layer in range(1, len(layers_shape))
        ]
        self.bias: np.ndarray = [
            np.zeros((layer, 1)) for layer in layers_shape[1:]
        ]
        self.activations: np.ndarray = [
            np.zeros((layer, batch_size)) for layer in layers_shape[1:]
        ]
        self.z: np.ndarray = [
            np.zeros((layer, batch_size)) for layer in layers_shape[1:]
        ]
        self.grads: np.ndarray = [np.zeros_like(W) for W in self.weights]
        self.grads_b: np.ndarray = [np.zeros_like(b) for b in self.bias]
        self.softmax_output: np.ndarray = np.zeros(
            (layers_shape[-1], batch_size))
        self.input_layer: np.ndarray = np.zeros(
            (layers_shape[0], batch_size))
        self.ground_truth: np.ndarray = np.zeros(
            (layers_shape[-1], batch_size))
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.current_batch_size = batch_size

    def __str__(self) -> str:
        msg = ""
        msg += f"Weights: ({len(self.weights)}, {len(self.weights[0])})\n"
        msg += f"Bias: ({len(self.bias)}, {len(self.bias[0])})\n"
        msg += f"Activations: ({len(self.activations)}, "
        msg += f"{len(self.activations[0])})\n"
        msg += f"Z: ({len(self.z)}, {len(self.z[0])})\n"
        msg += f"Grads: ({len(self.grads)}, {len(self.grads[0])})\n"
        msg += f"Bias Grads: ({len(self.grads_b)}, {len(self.grads_b[0])})\n"
        msg += f"Softmax Output: ({len(self.softmax_output)}, "
        msg += f"{len(self.softmax_output[0])})\n"
        msg += f"Input Layer: ({len(self.input_layer)}, "
        msg += f"{len(self.input_layer[0])})\n"
        msg += f"Ground Truth: ({len(self.ground_truth)}, "
        msg += f"{len(self.ground_truth[0])})\n"
        msg += f"Learning Rate: {self.learning_rate}\n"
        msg += f"Batch_size: {self.batch_size}"
        msg += f"Current batch_size: {self.current_batch_size} / {self.batch_size}"
        return msg

    class SizeDoNotMatch(Exception):
        def __init__(
            self,
            val1: str,
            val2: str,
            shape1: tuple = None,
            shape2: tuple = None
        ):
            msg = f"Shape mismatch: {val1} and {val2} size do not match."
            if shape1 and shape2:
                msg += f"Shape: {shape1} | {shape2}"
            super().__init__(msg)

    def get_layer(self, indice: int) -> tuple[list[float], list[float]]:
        """Return first weights then bias"""
        return self.weights[indice], self.bias[indice]

    def softmax(self, output_layer):
        """Output activation function"""
        exps = np.exp(output_layer - np.max(output_layer, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def sigmoid(self, z: np.ndarray):
        """Compression function avec clip pour eviter les overflow (exp)"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500))) 

    def partial_derivative_sigmoid(self, z: np.ndarray):
        """Compute and return the partial derivative of z sigmoid"""
        sigmoid: np.ndarray = self.sigmoid(z)
        return sigmoid * (1 - sigmoid)

    def compute_activation_layer(
        self, w: np.ndarray, prev_a: np.ndarray, bias: np.ndarray
    ):
        if w.shape[1] != prev_a.shape[0]:
            raise self.SizeDoNotMatch("weights", "previous activations")
        if len(w) != len(bias):
            raise self.SizeDoNotMatch("weights", "bias")
        return w @ prev_a + bias

    def compute_activation(self):
        """Compute activation for the input given"""
        for idx in range(len(self.activations) - 1):
            if idx == 0:
                prev_activations: np.ndarray = self.input_layer
            else:
                prev_activations: np.ndarray = self.activations[idx-1]
            W, b = self.get_layer(idx)
            self.z[idx]: np.ndarray = self.compute_activation_layer(
                W, prev_activations, b)
            self.activations[idx] = self.sigmoid(self.z[idx])
        # Last activations
        prev_activations = self.activations[-2]
        output_activation = self.compute_activation_layer(
            self.weights[-1], prev_activations, self.bias[-1]
        )
        self.softmax_output = self.softmax(output_activation)

    def cross_entropy(self) -> float:
        """Loss function"""
        epsilon = 1e-15
        softmax_clipped = np.clip(self.softmax_output, epsilon, 1 - epsilon)
        return -np.sum(self.ground_truth * np.log(softmax_clipped)) / self.current_batch_size

    def gradient_descent(self) -> None:
        """Weights update after backpropagation"""
        for idx in range(len(self.weights)):
            self.weights[idx] -= self.learning_rate * self.grads[idx]
            self.bias[idx] -= self.learning_rate * self.grads_b[idx]

    def backpropagation(self) -> None:
        """

        """
        for idx in range(len(self.grads) - 1, -1, -1):
            # Pour la couche la plus haute on calcul la variation du cout par rapoort au activation
            # -> (a(L) - y) ou a(L) represente les activations de la derniere couche (softmax) et y les ground truths
            # Ici on calcul dCo/da^L avec en sortie une cross entropy + softmax qui se calcul -> a^L - y
            if idx == len(self.grads) - 1:
                tmp_grads_b: np.ndarray = \
                    self.softmax_output - self.ground_truth
            
            # Pour les autres couche on calcul la variation des activations par rapport aux sommes ponderees
            # -> o'(z(L)) ou o' est la fonction d'activation et z(L) les sommes ponderees de la couche L
            else:
                w: np.ndarray = self.weights[idx + 1]
                if w.shape[0] != len(tmp_grads_b):
                    raise self.SizeDoNotMatch(
                        "weights", "upper grad", w.shape, len(tmp_grads_b))
                propagated_error: np.ndarray = w.T @ tmp_grads_b
                tmp_grads_b = propagated_error * \
                    self.partial_derivative_sigmoid(self.z[idx])
                if len(self.grads_b[idx]) != len(self.activations[idx]):
                    raise self.SizeDoNotMatch(
                        "bias gradient", "previous activation")
            # Division par batch_size pour avoir une moyenne des exemple du batch
            if idx == 0:
                prev_activations = self.input_layer
            else:
                prev_activations = self.activations[idx - 1]
            self.grads[idx] = (
                tmp_grads_b @ prev_activations.T
            ) / self.current_batch_size
            # Moyenne sur les lignes donc moyennes du batch
            # (les colonnes sont les exemples,
            # les lignes representes les neuronnes actuels)
            self.grads_b[idx] = np.mean(tmp_grads_b, axis=1, keepdims=True)
        self.gradient_descent()

    def train(self, input: np.ndarray, label: np.ndarray) -> float:
        if input.shape[0] != self.input_layer.shape[0]:
            raise self.SizeDoNotMatch("input", "input_layer")
        if label.shape[0] != self.ground_truth.shape[0]:
            raise self.SizeDoNotMatch("label", "ground_truth")

        self.current_batch_size = input.shape[1]
        # Pour le dernier batch ou cas ou il est incomplet
        self.input_layer[:, :self.current_batch_size] = input
        self.ground_truth[:, :self.current_batch_size] = label
        print("=================")
        print("Compute Activation...")
        self.compute_activation()
        loss = self.cross_entropy()
        print(f"Cross-entropy loss: {loss:.8f}")
        #print(f"Res: {self.human_readable_output()}")
        print("=================")
        print("Backpropagation...")
        self.backpropagation()
        return loss
    
    def human_readable_output(self):
        res = []
        print(self.softmax_output)
        print(self.ground_truth)
        for out, g_t in zip(self.softmax_output, self.ground_truth):
            if g_t == 1:
                if out > 0.5:
                    res.append(True)
                else:
                    res.append(False)
            else:
                if out < 0.5:
                    res.append(True)
                else:
                    res.append(False)
        return res

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
