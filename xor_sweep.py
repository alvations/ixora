from itertools import product
from collections import Counter 

import time

import random
random.seed(100)

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
from torch import optim
use_cuda = torch.cuda.is_available()

# Activation functions.
from torch.nn import ReLU, ReLU6, ELU, SELU, LeakyReLU
from torch.nn import Hardtanh, Sigmoid, Tanh, LogSigmoid
from torch.nn import Softplus, Softshrink, Tanhshrink, Softmin
from torch.nn import Softmax, LogSoftmax # Softmax2d


# Loss functions.
from torch.nn import L1Loss, MSELoss # NLLLoss, CrossEntropyLoss
from torch.nn import PoissonNLLLoss, KLDivLoss, BCELoss
from torch.nn import BCEWithLogitsLoss, HingeEmbeddingLoss # MarginRankingLoss
from torch.nn import SmoothL1Loss, SoftMarginLoss # MultiLabelMarginLoss, CosineEmbeddingLoss, 
from torch.nn import MultiLabelSoftMarginLoss # MultiMarginLoss, TripletMarginLoss

# Optimizers.
from torch.optim import Adadelta, Adagrad, Adam, Adamax # SparseAdam
from torch.optim import ASGD, RMSprop, Rprop # LBFGS

Activations = [ReLU, ReLU6, ELU, SELU, LeakyReLU, 
                Hardtanh, Sigmoid, Tanh, LogSigmoid,
                Softplus, Softshrink, Tanhshrink, Softmin, 
                Softmax, LogSoftmax]

Criterions = [L1Loss, MSELoss,
              PoissonNLLLoss, KLDivLoss, BCELoss,
              BCEWithLogitsLoss, HingeEmbeddingLoss,
              SmoothL1Loss, SoftMarginLoss,
              MultiLabelSoftMarginLoss]

Optimizers = [Adadelta, Adagrad, Adam, Adamax,
             ASGD, RMSprop, Rprop]


X = xor_input = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = xor_output = np.array([[0,1,1,0]]).T

# Converting the X to PyTorch-able data structure.
X_pt = Variable(FloatTensor(X))
X_pt = X_pt.cuda() if use_cuda else X_pt
# Converting the Y to PyTorch-able data structure.
Y_pt = Variable(FloatTensor(Y), requires_grad=False)
Y_pt = Y_pt.cuda() if use_cuda else Y_pt

# Use FloatTensor.shape to get the shape of the matrix/tensor.
num_data, input_dim = X_pt.shape
num_data, output_dim = Y_pt.shape

learning_rate = [1.0, 0.5, 0.3, 0.1, 0.05, 0.03]
hidden_dims = [5, 2]
num_epochs = [10, 50, 100, 1000, 2000, 3000, 5000]
num_experiments = 100


for hidden_dim, lr, epochs, Activation, Criterion, Optimizer in product(hidden_dims, learning_rate, num_epochs, Activations, Criterions, Optimizers):
    all_results=[]
    start = time.time()
    #print(Activation, Criterion, Optimizer)
    for _ in range(num_experiments):
        model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                              Activation(), 
                              nn.Linear(hidden_dim, output_dim),
                              nn.Sigmoid())
        model = model.cuda() if use_cuda else model
        criterion = Criterion()
        optimizer = Optimizer(model.parameters(), lr=lr)
        
        for _e in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_pt)
            loss_this_epoch = criterion(predictions, Y_pt)
            loss_this_epoch.backward()
            optimizer.step()
            ##print(_e, [float(_pred) for _pred in predictions], list(map(int, Y_pt)), loss_this_epoch.data[0])

        x_pred = [int(model(_x) > 0.5) for _x in X_pt]
        y_truth = list([int(_y[0]) for _y in Y_pt])
        all_results.append([x_pred == y_truth, x_pred, loss_this_epoch.data[0]])

    tf, outputsss, losses__ = zip(*all_results)
    a_name, c_name, o_name = str(Activation).rpartition('.')[2][:-2], str(Criterion).rpartition('.')[2][:-2], str(Optimizer).rpartition('.')[2][:-2]
    print('\t'.join(map(str, [a_name, c_name, o_name, hidden_dim, lr, epochs, Counter(tf)[True], Counter(tf)[False], time.time()-start ])), flush=True)
