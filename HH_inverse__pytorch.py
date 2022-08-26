import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import time

torch.manual_seed(123456)                                                          # Seed for torch
np.random.seed(123456)                                                             # Seed for np
q1 = np.round(np.random.uniform(115, 123),1)                                      # Random point from distribution
q2 = np.round(np.random.uniform(32, 37),1)                                      # Random point from distribution
q3 = np.round(np.random.uniform(0.2, 0.5),1)                                      # Random point from distribution
device = torch.device('cpu')


#  Deep Neural Network
class DNN(nn.Module):
    def __init__(self, layers):

        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh                                                  # Activation function for each layer
        layer_list = list()

        for i in range(self.depth - 1):
            layer_list.append(('Linear_Layer_%d'% i, nn.Linear(layers[i], layers[i + 1])))      # Linear layer
            layer_list.append(('Tanh_Layer_%d' % i, self.activation()))                         # Activation layer

        layer_list.append(('Layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1])))   # Append solution layer to list
        layerDict = OrderedDict(layer_list)                                                     # Recalls the order of entries
        self.layers = nn.Sequential(layerDict)                                                  # Sequential container

    # Forward pass of the network to predict y
    def forward(self, x):
        out = self.layers(x)
        return out

# Physics Informed Neural Network
class PINNs():
    def __init__(self, t, v,m,h,n, layers):

                     # Interior x
        self.t = torch.tensor(t, requires_grad=True).float().to(device)                 # Interior t

        self.v = torch.tensor(v, requires_grad=True).float().to(device)                                         # Exact rho
        self.m = torch.tensor(m).float().to(device)                                             # Exact u
        self.h = torch.tensor(h).float().to(device)                                             # Exact p
        self.n = torch.tensor(n).float().to(device)                                             # Exact E

        self.g1 = torch.tensor([q1], requires_grad=True).to(device)                           # Define gamma
        self.g2 = torch.tensor([q2], requires_grad=True).to(device)
        self.g3 = torch.tensor([q3], requires_grad=True).to(device)
        
        # Define gamma

        # Register gamma as parameter to optimize
        self.g1 = nn.Parameter(self.g1)
        self.g2 = nn.Parameter(self.g2)
        self.g3 = nn.Parameter(self.g3)
        # Register gamma
        # Register gamma
        self.dnn = DNN(layers).to(device)                                                       
        # DNN
        self.dnn.register_parameter('g1', self.g1)
        self.dnn.register_parameter('g2', self.g2)
        self.dnn.register_parameter('g3', self.g3)
        
        # Optimizer - Limited Memory Broyden–Fletcher–Goldfarb–Shannon Algorithm 
        self.optimizer = torch.optim.LBFGS(              # LBFGS
            self.dnn.parameters(),                       # Optimize theta and gamma
            lr=1e-1,                                      # Learning Rate
            max_iter=1000,                               # Default max # of iterations per optimization step                                                                    #
            tolerance_grad=1e-20,                        # Default termination tolerance on first order optimality
            tolerance_change=1e-20,                      # Default termination tolerance on function value/parameter changes
            history_size=1000
        )
      
        self.iter = 0                                    # Initialize iterations

    # Neural network solution y = [v(x,t) , m(x,t), h(x,t), n(x,t)]
    def net_y(self, t):
        y = self.dnn(t)
        return y

    # General Loss Function
    def loss_func(self):
        y_pred = self.net_y(self.t)    
        v_nn, m_pred, h_pred, n_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3] # NN_{rho}, NN_{u}, NN_{p}

        # Reshape data
        m_pred = m_pred.reshape(len(m_pred), 1) # 和m h n  loss
        h_pred = h_pred.reshape(len(h_pred), 1)
        n_pred = n_pred.reshape(len(n_pred), 1)
        
        v_nn = v_nn.reshape(len(v_nn), 1)

        v_pred = 10.0- (self.g1 * m_pred**3 * h_pred *(v_nn-50.0))-\
                 (self.g2 * n_pred**4 * (v_nn-77.0))-(self.g3 * (v_nn-54.387))

        # Total Loss
       
        loss = torch.mean((self.m - m_pred) ** 2) + torch.mean((self.h - h_pred) ** 2) + \
                torch.mean((self.n - n_pred) ** 2) + torch.mean(((self.v - v_pred)) ** 2)
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        # if self.iter%101==0:
        # print("iter: ",self.iter)
        print(
        'Loss: %.3f, g1_PINNs: %.5f ,g2_PINNs: %.5f,g3_PINNs: %.5f ' %
        (
            loss.item(),

            self.g1.item(),
            self.g2.item(),
            self.g3.item()
            )
        )
        return loss

    # Train network through minimization of loss function w/r to theta and gamma
    def train(self, nIter):
        self.dnn.train()
        # Backward and optimize
        self.optimizer.step(self.loss_func)

data = np.load('g_Na=120_g_K=36_g_L=0.3.npz')
t = data['t']       # Partitioned time coordinates
Exact_v = data['v']
Exact_m = data['m']
Exact_h = data['h']
Exact_n = data['n']

# Define PINNs Model
# Initialization
layers = [1]+[20]*3+[4]

model = PINNs(t=t, v=Exact_v,m=Exact_m,h=Exact_h,n=Exact_n, layers=layers)
tic = time.time()
model.train(0)
toc = time.time()
print(f'total training time: {toc - tic}')  
print("initial value： %.3f,%.3f,%.3f,"% (q1,q2,q3))
