import numpy as np
from typing import Tuple
from .base import Module, Optimizer


class SGD(Optimizer):
    """
    Optimizer implementing stochastic gradient descent with momentum
    """
    def __init__(self, module: Module, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param momentum: momentum coefficient (alpha)
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]

        for param, grad, m in zip(parameters, gradients, self.state['m']):
            """
            your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
              - update momentum variable (m)
              - update parameter variable (param)
            hint: consider using np.add(..., out=m) for in place addition,
              i.e. we need to change original array, not its copy
            """
            
            if self.weight_decay != 0:
                grad+= self.weight_decay* param
                
            if self.momentum != 0:
                m *= self.momentum
                m += grad
      
                
                if self.nesterov == True:
                    grad = grad + self.momentum * m
                else:
                    grad = m

            param -= self.lr * grad

            pass


class Adam(Optimizer):
    """
    Optimizer implementing Adam
    """
    def __init__(self, module: Module, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        """
        :param module: neural network containing parameters to optimize
        :param lr: learning rate
        :param betas: Adam beta1 and beta2
        :param eps: Adam eps
        :param weight_decay: weight decay (L2 penalty)
        """
        super().__init__(module)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.parameters_grad()
        if 'm' not in self.state:
            self.state['m'] = [np.zeros_like(param) for param in parameters]
            self.state['v'] = [np.zeros_like(param) for param in parameters]
            self.state['t'] = 0

        self.state['t'] += 1
        t = self.state['t']
        for param, grad, m, v in zip(parameters, gradients, self.state['m'], self.state['v']):
            """
            your code here ｀、ヽ｀、ヽ(ノ＞＜)ノ ヽ｀☂｀、ヽ
              - update first moment variable (m)
              - update second moment variable (v)
              - update parameter variable (param)
            hint: consider using np.add(..., out=m) for in place addition,
              i.e. we need to change original array, not its copy
            """
            wd = self.weight_decay
            beta1 = self.beta1
            beta2 = self.beta2
            eps = self.eps
            lr = self.lr
            
            if wd != 0:
                grad = grad +wd *param      
    
            m *= beta1
            m += (1- beta1)* grad
            v *= beta2
            v += (1- beta2) * (grad**2)

            m_hat = 0
            v_hat = 0
            m_hat = m/(1 -(beta1**t))
            v_hat = v/(1 -(beta2**t))
    

            param -= lr*m_hat/(np.sqrt(v_hat)+eps)

            
            pass
