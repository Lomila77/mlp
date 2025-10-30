import pickle
import numpy as np
from configs.config import MODEL_PATH


class Model:
    input_layer: np.ndarray
    output_layer: np.ndarray

    def __init__(
        self,
        layers_shape: list[int],
        batch_size: int = 1,
        learning_rate: float = 0.7,
        load_last_checkpoint: bool = False
    ) -> None:
        """Initialise the model.

        Args:
            layers_shape (list[int]): The shape of the model, it is adaptable
            batch_size (int): The size of the batch -> How many example my model can ingest at the same time
            learning_rate (float): The step of gradients update
        """
        if load_last_checkpoint:
            with open(MODEL_PATH, 'rb') as f:
                model_checkpoint = pickle.load(f)
            self.shape = model_checkpoint['shape']
            self.weights = model_checkpoint['weights']
            self.bias = model_checkpoint['bias']
        else:
            self.shape = layers_shape
            self.weights: np.ndarray = [
                np.random.randn(
                    self.shape[layer], self.shape[layer-1]
                ) for layer in range(1, len(self.shape))
            ]
            self.bias: np.ndarray = [
                np.zeros((layer, 1)) for layer in self.shape[1:]
            ]
        self.activations: np.ndarray = [
            np.zeros((layer, batch_size)) for layer in self.shape[1:]
        ]
        self.z: np.ndarray = [
            np.zeros((layer, batch_size)) for layer in self.shape[1:]
        ]
        self.grads: np.ndarray = [np.zeros_like(W) for W in self.weights]
        self.grads_b: np.ndarray = [np.zeros_like(b) for b in self.bias]
        self.softmax_output: np.ndarray = np.zeros(
            (self.shape[-1], batch_size))
        self.input_layer: np.ndarray = np.zeros(
            (self.shape[0], batch_size))
        self.ground_truth: np.ndarray = np.zeros(
            (self.shape[-1], batch_size))
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.current_batch_size = batch_size
        self.loss = []
        self.v_loss = []
        self.accuracy_score = []
        self.precision_score = []
        self.recall_score = []
        self.f1_score = []
        print(self)

    def __str__(self) -> str:
        msg = "MODEL INFORMATION:\n"
        msg += f"Shape: {self.shape}\n"
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
        msg += f"Batch_size: {self.batch_size}\n"
        msg += f"Current batch_size: {self.current_batch_size} / {self.batch_size}\n"
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
        """Return the layer for the given index"""
        return self.weights[indice], self.bias[indice]

    def softmax(self, output_layer: np.ndarray):
        """The output activation function. 
        It tells probabilistic distribution of the output.

        Args:
            output_layer (np.ndarray): The output layer (z)

        Returns:
            np.ndarray: The probabilistic distribution
        """
        exps = np.exp(output_layer - np.max(
            output_layer, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def sigmoid(self, z: np.ndarray):
        """The activation function. It compress the score between 0 and 1.
        "clip" is use to avoid overflow (because of using exp).

        Returns:
            np.ndarray: Return activations
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500))) 

    def partial_derivative_sigmoid(self, z: np.ndarray):
        """Compute the partial derivative of sigmoid(z).
        Needed to compute the derivative of activation on ponderate sum,
        because i want know what is the variation between a and z.
        We know, from z to a, we have the sigmoid function,
        so the derivative of the sigmoid is the derivative of a by z.

        Returns:
            np.ndarray: The partial derivative sigmoid of the layer.
        """
        sigmoid: np.ndarray = self.sigmoid(z)
        return sigmoid * (1 - sigmoid)

    def compute_ponderate_sum(
        self, w: np.ndarray, prev_a: np.ndarray, bias: np.ndarray
    ):
        """Compute activation for only one layer"""
        if w.shape[1] != prev_a.shape[0]:
            raise self.SizeDoNotMatch("weights", "previous activations")
        if len(w) != len(bias):
            raise self.SizeDoNotMatch("weights", "bias")
        return w @ prev_a + bias

    def forward_pass(self):
        """Compute activations of the networks.
        Given the input layer, we compute the activation layer per layer.
        At the end the last activation compute is a softmax.
        Feedforward: the data flows from the input layer to the output layer.
        """
        for idx in range(len(self.activations) - 1):
            if idx == 0:
                prev_activations: np.ndarray = self.input_layer
            else:
                prev_activations: np.ndarray = self.activations[idx-1]
            W, b = self.get_layer(idx)
            self.z[idx]: np.ndarray = self.compute_ponderate_sum(
                W, prev_activations, b)
            self.activations[idx] = self.sigmoid(self.z[idx])
        # Last activations
        prev_activations = self.activations[-2]
        output_activation = self.compute_ponderate_sum(
            self.weights[-1], prev_activations, self.bias[-1]
        )
        self.softmax_output = self.softmax(output_activation)

    def cross_entropy(self) -> float:
        """Loss function"""
        epsilon = 1e-15
        softmax_clipped = np.clip(self.softmax_output, epsilon, 1 - epsilon)
        entropy = -np.sum(self.ground_truth * np.log(
            softmax_clipped)) / self.current_batch_size
        return entropy

    def gradient_descent(self) -> None:
        """Weights update after backpropagation"""
        for idx in range(len(self.weights)):
            self.weights[idx] -= self.learning_rate * self.grads[idx]
            self.bias[idx] -= self.learning_rate * self.grads_b[idx]

    def backpropagation(self) -> None:
        """Compute gradients to update the weights.
        This algorythm start from the output layer and follow:
        dCo/dW^L = dz^L/dW^L * da^L/dz^L * dCo/da^L
    
        This first step compute the cost variation for the activation:
        -> dCo/da^L:
        * (a^L - y) where "a" is the softmax output and "y" the labels.
        * This compute combine implicitly the cross-entropy loss and the
        softmax output.

        For other layer we compute activation variation for the ponderate sum:
        -> da^L/dz^L:
        * o'(z^L) where "o'" is the partial derivative sigmoid and "z" the 
        ponderate sum.

        The last compute is very simple, by following the derivative rule,
        with this calcul:
        -> dz^L/dW^L
        * We obtain a^L-1 
        (f(W^L) = z^L = W^L * a^L-1 + b^L = 1 * a^L-1 + 0 = a^L-1)

        And then the last compute:
        -> dCo/dW^L
        * With "tmp_grads_b" which are da^L/dz^L * dCo/da^L
        * With "prev_activation wich are dz^L/dW^L
        * We obtain the gradients for the given layer.
        
        TO NOTE: I process inputs on batches:
        self.grads_b[idx] = np.mean(tmp_grads_b, axis=1, keepdims=True)
        -> Its the mean of all batches (columns->axis=1), its like compression.

        Raises:
            self.SizeDoNotMatch: If size missmatch we trow an error
        """
        for idx in range(len(self.grads) - 1, -1, -1):
            if idx == len(self.grads) - 1:
                tmp_grads_b: np.ndarray = \
                    self.softmax_output - self.ground_truth
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
            if idx == 0:
                prev_activations = self.input_layer
            else:
                prev_activations = self.activations[idx - 1]
            self.grads[idx] = (
                tmp_grads_b @ prev_activations.T
            ) / self.current_batch_size
            self.grads_b[idx] = np.mean(tmp_grads_b, axis=1, keepdims=True)
        self.gradient_descent()

    def train_step(self, input: np.ndarray, label: np.ndarray) -> float:
        """Train Step in the training process.
        Note that input and label can be incomplete because of
        processing per batches. It explain the slicing.

        Args:
            input (np.ndarray): Input batches
            label (np.ndarray): Ground truth batches

        Raises:
            self.SizeDoNotMatch: If size doesn't match the feature number.

        Returns:
            float: Loss results
        """
        if input.shape[0] != self.input_layer.shape[0]:
            raise self.SizeDoNotMatch("input", "input_layer")
        if label.shape[0] != self.ground_truth.shape[0]:
            raise self.SizeDoNotMatch("label", "ground_truth")
        self.current_batch_size = input.shape[1]
        self.input_layer[:, :self.current_batch_size] = input
        self.ground_truth[:, :self.current_batch_size] = label
        self.forward_pass()
        loss = self.cross_entropy()
        self.backpropagation()
        return loss

    def validation_step(self, input: np.ndarray, label: np.ndarray) -> float:
        """The validation step in the training process.
        Note that input and label can be incomplete because of
        processing per batches. It explain the slicing.


        Args:
            input (np.ndarray): Input batches.
            label (np.ndarray): Ground truth batches.

        Raises:
            self.SizeDoNotMatch: If size doesn't match the feature number.

        Returns:
            float: The loss results.
        """
        if input.shape[0] != self.input_layer.shape[0]:
            raise self.SizeDoNotMatch("input", "input_layer")
        if label.shape[0] != self.ground_truth.shape[0]:
            raise self.SizeDoNotMatch("label", "ground_truth")

        self.current_batch_size = input.shape[1]
        self.input_layer[:, :self.current_batch_size] = input
        self.ground_truth[:, :self.current_batch_size] = label
        self.forward_pass()
        loss = self.cross_entropy()
        acc, pre, rec, f1 = self.evaluation()
        return {
            "loss": loss,
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1
        }

    def train(
        self,
        train_dataset: np.ndarray,
        val_dataset: np.ndarray,
        epochs: int
    ) -> tuple[list, list]:
        """Training process.
        Display epochs and entropy loss.
        The accuracy, precision and recall scores are automaticaly compute.

        Args:
            train_dataset (np.ndarray): Train dataset.
            val_dataset (np.ndarray): Validation dataset.
            epochs (int): The number of iteration on both dataset.

        Returns:
            dict: Returns loss, validation loss, accuracy, precision and recall
        """
        try:
            current_epoch = 0
            for epoch in range(epochs):
                current_epoch = epoch
                print("====================================")
                print(f"Epochs: {epoch + 1}")
                print("=================")
                print("TRAIN:")
                loss = []
                for input, label in train_dataset:
                    loss.append(self.train_step(input, label))
                self.loss.append(np.mean(loss))
                print(f"Cross-entropy loss: {self.loss[-1]:.8f}")
                print("=================")
                print("VALIDATION:")
                v_loss = []
                acc = []
                pre = []
                rec = []
                f1 = []
                for input, label in val_dataset:
                    metrics = self.validation_step(input, label)
                    v_loss.append(metrics["loss"])
                    acc.append(metrics["accuracy"])
                    pre.append(metrics["precision"])
                    rec.append(metrics["recall"])
                    f1.append(metrics["f1"])
                self.accuracy_score.append(np.mean(acc))
                self.precision_score.append(np.mean(pre))
                self.recall_score.append(np.mean(rec))
                self.f1_score.append(np.mean(f1))
                self.v_loss.append(np.mean(v_loss))
                print(f"Cross-entropy loss: {self.v_loss[-1]:.8f}")
                print("====================================\n\n")
            return {
                "epochs": epochs,
                "loss": self.loss,
                "v_loss": self.v_loss,
                "accuracy": self.accuracy_score,
                "precision": self.precision_score,
                "recall": self.recall_score,
                "f1": self.f1_score
            }
        except KeyboardInterrupt:
            return {
                "epochs": current_epoch,
                "loss": self.loss,
                "v_loss": self.v_loss,
                "accuracy": self.accuracy_score,
                "precision": self.precision_score,
                "recall": self.recall_score,
                "f1": self.f1_score
            }
        except Exception as e:
            raise e

    def accuracy(self, preds, truths):
        return np.mean(preds == truths)

    def precision(self, preds, truths):
        tp = np.sum((preds == 1) & (truths == 1))
        fp = np.sum((preds == 1) & (truths == 0))
        return tp / (tp + fp) if tp + fp > 0 else 0

    def recall(self, preds, truths):
        tp = np.sum((preds == 1) & (truths == 1))
        fn = np.sum((preds == 0) & (truths == 1))
        return tp / (tp + fn) if tp + fn > 0 else 0

    def f1(self, pre, rec):
        return 2 * (pre * rec) / (pre + rec) if pre + rec > 0 else 0

    def evaluation(self) -> tuple[float, float, float, float]:
        predictions = np.argmax(self.softmax_output, axis=0)
        ground_truths = np.argmax(self.ground_truth, axis=0)

        acc = self.accuracy(predictions, ground_truths)
        pre = self.precision(predictions, ground_truths)
        rec = self.recall(predictions, ground_truths)
        f1_ = self.f1(pre, rec)

        return acc, pre, rec, f1_



    def human_readable_output(self):
        if self.softmax_output[0] > self.softmax_output[1]:
            return "M"
        else:
            return "B"

    def predict(self, input: np.ndarray):
        if input.shape[0] != self.input_layer.shape[0]:
            raise self.SizeDoNotMatch("input", "input_layer")
        self.current_batch_size = 1
        self.input_layer[:, :self.current_batch_size] = input
        self.forward_pass()
        answer: str = self.human_readable_output()
        return answer

    def save_model(self):
        """Save weights and bias into json file"""
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'shape': self.shape,
                'weights': self.weights,
                'bias': self.bias
            }, f)
