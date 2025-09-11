import json


class Model:
    weights: list[list[float]]
    bias: list[list[float]]
    activations: list[list[float]]
    gradients: list[list[float]]
    output: float

    def __init__(self, weigths: list[list[float]], bias: list[list[float]]) -> None:
        self.weights = weigths
        self.bias = bias

    def get_layer(self, indice: int) -> tuple[list[float], list[float]]:
        return self.weights[indice], self.bias[indice]

    def softmax(self):
        """Output activation function"""
        pass

    def sigmoid(self, z: float):
        """Compression function"""
        pass

    def partial_derivative_sigmoid(self, z: float):
        """Compute and return the partial derivative of z sigmoid"""
        sigmoid: float = self.sigmoid(z)
        return sigmoid * (1 - sigmoid)

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

    def cross_entropy(self) -> float:
        """Loss function"""
        pass

    def gradient(self) -> float:
        pass

    def transpose(self, matrix: list[list[float]]) -> list[list[float]]:
        pass

    # TODO: prendre en compte le softmax de fin
    def backpropagation(self, idx: int) -> float:
        """
        Recursive Function which travels along layers,
        from the end to the beginning.
        Set self.gradients
        """
        upper_gradient: float = sum(self.gradients[idx + 1])
        curr_gradient: list[float] = []
        for idx in range(len(self.gradients) - 1, -1, -1):
            propagated_error: float = sum(self.transpose(self.weights[idx + 1])) * upper_gradient
            for weight, bias in zip(self.weights[idx], self.bias[idx]):

            
            self.gradients[idx]
        pass

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
