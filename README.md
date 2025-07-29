# Richard-cases
#case 1 
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import pandas as pd
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(2,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## 
        return output
        ### (2) Model
net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters())
## PDE as loss function. Thus would use the network which we call as u_theta
def f(x,t, net):
    u = net(x,t) # the dependent variable u is given by the network based on independent variables x,t
  
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx= torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]


    pde = u_t+u*u_x-u_xx
    return pde
    #1st BC


x_bc1 = np.random.uniform(low=0.00, high=5.00, size=(1500,1))
t_bc1 = np.zeros((1500,1))
u_bc1 =(1/2+1/2*np.tanh(-x_bc1/4))
### (3) Training / Fitting
iterations = 50000
previous_validation_loss = 99999999.0
training_loss_values = []

for epoch in range(iterations):
    optimizer.zero_grad() # to make the gradients zero

    # Loss based on boundary conditions
    pt_x_bc1 = Variable(torch.from_numpy(x_bc1).float(), requires_grad=False).to(device)
    pt_t_bc1 = Variable(torch.from_numpy(t_bc1).float(), requires_grad=False).to(device)
    pt_u_bc1 = Variable(torch.from_numpy(u_bc1).float(), requires_grad=False).to(device)

    net_bc_out = net(pt_x_bc1, pt_t_bc1) # output of u(x,t)
    mse_u1 = mse_cost_function(net_bc_out, pt_u_bc1)

    # Loss based on PDE
    x_collocation = np.random.uniform(low=0.0, high=5.00, size=(7000,1))
    t_collocation = np.random.uniform(low=0.0, high=10, size=(7000,1))
    all_zeros = np.zeros((7000,1))


    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    f_out = f(pt_x_collocation, pt_t_collocation, net) # output of f(x,t)
    mse_f = mse_cost_function(f_out, pt_all_zeros)

    # Combining the loss functions
    loss = mse_u1+ mse_f


    loss.backward() # This is for computing gradients using backward propagation
    optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
      '''
    	print(epoch,"Traning Loss:",loss.data)
      '''

    training_loss_values.append(loss.item())  # Store the MSE loss value

    if epoch % 10000 == 0:
        print(epoch, "Training Loss:", loss.item())


  # case 2

  import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import pandas as pd
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(2,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## 
        return output
        ### (2) Model
net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters())
## PDE as loss function. Thus would use the network which we call as u_theta
def f(x,t, net):
    u = net(x,t) # 
    
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx= torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]


    pde = u_t+u**2*u_x-u_xx
    return pde
    #1st IC


x_bc1 = np.random.uniform(low=0.00, high=10, size=(1000,1))
t_bc1 = np.zeros((1000,1))
u_bc1 =(1/2+1/2*np.tanh(-x_bc1/3))**(1/2)
### (3) Training / Fitting
iterations = 50000
previous_validation_loss = 99999999.0
training_loss_values = []

for epoch in range(iterations):
    optimizer.zero_grad() # to make the gradients zero

    # Loss based on boundary conditions
    pt_x_bc1 = Variable(torch.from_numpy(x_bc1).float(), requires_grad=False).to(device)
    pt_t_bc1 = Variable(torch.from_numpy(t_bc1).float(), requires_grad=False).to(device)
    pt_u_bc1 = Variable(torch.from_numpy(u_bc1).float(), requires_grad=False).to(device)

    net_bc_out = net(pt_x_bc1, pt_t_bc1) # output of u(x,t)
    mse_u1 = mse_cost_function(net_bc_out, pt_u_bc1)

    # Loss based on PDE
    x_collocation = np.random.uniform(low=0.0, high=5, size=(5000,1))
    t_collocation = np.random.uniform(low=0.0, high=10, size=(5000,1))
    all_zeros = np.zeros((5000,1))


    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    f_out = f(pt_x_collocation, pt_t_collocation, net) # output of f(x,t)
    mse_f = mse_cost_function(f_out, pt_all_zeros)

    # Combining the loss functions
    loss = mse_u1+ mse_f


    loss.backward() # This is for computing gradients using backward propagation
    optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
      '''
    	print(epoch,"Traning Loss:",loss.data)
      '''

    training_loss_values.append(loss.item())  # Store the MSE loss value

    if epoch % 10000 == 0:
        print(epoch, "Training Loss:", loss.item())
