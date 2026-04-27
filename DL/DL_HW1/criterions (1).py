import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ

        return (((input - target)**2).sum()) / input.size

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2*(input - target)/ input.size
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ

        


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        logprobs = self.log_softmax.compute_output(input)

        if self.label_smoothing == 0:
            return - np.mean(logprobs[np.arange(len(input)), target])
        else:
            correct = logprobs[np.arange(target.size), target]      
            sum_all = np.sum(logprobs, axis=1)            
            loss = -(1- self.label_smoothing) * correct - (self.label_smoothing/ input.shape[1])* sum_all

            return np.mean(loss)
        
        # replace with your code ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ


    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """

        if self.label_smoothing == 0:
            logprobs = self.log_softmax.compute_output(input)   
            probs = np.exp(logprobs)                            
            grad = probs.copy()
            grad[np.arange(input.shape[0]), target] -= 1
            return grad/input.shape[0]

        else:
            logprobs = self.log_softmax.compute_output(input)
            probs = np.exp(logprobs)
            grad = probs.copy()
            
            grad[np.arange(input.shape[0]), target] -= (1 - self.label_smoothing)  
            grad -= self.label_smoothing/input.shape[1] 

            return grad/input.shape[0]

