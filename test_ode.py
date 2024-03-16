#!/hpc/home/zw122/miniconda3/envs/goujio/bin/python
# %%
import sys
sys.path.append('/hpc/home/zw122/tree_condsamp/LACE/tree-toy/')
sys.path.append('/hpc/home/zw122/tree_condsamp/boostPM_py')
import sampling
import importlib
import torch
import boostPM_py
import matplotlib.pyplot as plt
import rpy2.robjects as robj
import os


# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nsamp', type=int)
parser.add_argument('--beta_0', type=float)
parser.add_argument('--beta_1', type=float)
parser.add_argument('--rtol', type=float)
parser.add_argument('--atol', type=float)
parser.add_argument('--ode_T', type=float)
parser.add_argument('--ode_method', type=str)


args = parser.parse_args()
nsamp = args.nsamp 
beta_0 = args.beta_0
beta_1 = args.beta_1
rtol = args.rtol
atol = args.atol
prefix = "ode_gaussian" + "_nsamp_" + str(nsamp) + "_beta_0_" + str(beta_0) + "_beta_1_" + str(beta_1) + "_rtol_" + str(args.rtol) + "_atol_" + str(args.atol) + "_" + args.ode_method + "_T_" + str(args.ode_T) + "_"
print('/work/zw122/tree_condsamp/LACE/tree-toy/cache/'+prefix+'temp_s.pdf', flush=True)
path = '/work/zw122/tree_condsamp/LACE/tree-toy/cache/'+prefix+'temp_gs.pkl'
if os.path.exists(path):
    print('file exists', flush=True)
    exit()
if beta_1 <= beta_0:
    print('beta_1 <= beta_0', flush=True)
    exit()
# %%
# prepare training data and pretrained f and g
torch.manual_seed(1)
n = 1000
d_g = 2
Xtrain = torch.zeros((2*n, d_g)).double()
c = torch.cat((torch.zeros((n,1)),torch.ones((n,1))), dim = 0).double()
Xtrain[(c==1).flatten(),:] = torch.randn(n, d_g).double() @ torch.tensor([[1, 0.3], [0.3, 1]]).double()
Xtrain[(c==0).flatten(),:] = (torch.randn(n, d_g) + 8.).double() @ torch.tensor([[1, -0.3], [-0.3, 1]]).double()
Xtrain = Xtrain.double()
#-------generator---------
out = boostPM_py.boosting(Xtrain)
robj.r('saveRDS')(out, '/hpc/home/zw122/tree_condsamp/LACE/tree-toy/pretrained/out.rds')

# %%
# plt.scatter(Xtrain[(c==0).flatten(),0],Xtrain[(c==0).flatten(),1], label = 'Xtrain (0)', alpha = 0.2)
# plt.scatter(Xtrain[(c==1).flatten(),0],Xtrain[(c==1).flatten(),1], label = 'Xtrain (1)', alpha = 0.2)
# plt.legend();

# %%
import torch.nn as nn
import torch.optim as optim

hidden_size1 = 3
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1).double()
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2).double()
        self.fc3 = nn.Linear(hidden_size1, d_g).double()

    def forward(self, x):
        return self.fc3(torch.relu(self.fc1(x)))

# Hyperparameters
input_size = d_g  # Size of input features
num_epochs = 200
learning_rate = 0.01
batch_size = 100

# Define the model
model = BinaryClassifier(input_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Prepare the data
labels = c.long().flatten()  # Binary labels

# Create a DataLoader for batch training
dataset = torch.utils.data.TensorDataset(Xtrain, labels)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        # Forward pass
        outputs = model(inputs)
        probs = nn.functional.softmax(outputs, dim=1)

        # Compute the loss
        loss = criterion(probs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate the trained model
with torch.no_grad():
    outputs = model(Xtrain)
    predicted_probs = nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
    _, predicted_labels = torch.max(predicted_probs, 1)  # Convert probabilities to predicted labels
    accuracy = (predicted_labels == labels).float().mean()
    print(f'Accuracy: {accuracy.item()}')


# %%
ccf = sampling.CCF(x_space="toy_i", latent_dim = d_g, n_classes = 2)
ccf.f = model
sample_q = sampling._sample_q_dict['sample_q_ode']
ode_kwargs = {'every_n_plot': 100, 'save_path': True, 'batch_size': 500, 'latent_dim': d_g, 'rtol': rtol, 'atol': atol, 'device': torch.device('cpu'), 'method': args.ode_method, 'use_adjoint': False, 'beta_0':beta_0, 'beta_1':beta_1, 'T':args.ode_T}



# %%
y = torch.cat((torch.ones(nsamp), torch.zeros(nsamp)), axis = 0).to(torch.long)
s = sample_q(ccf = ccf, y = y, **ode_kwargs)

# %%
gs = ccf.g(s).detach()
import pickle
with open('/work/zw122/tree_condsamp/LACE/tree-toy/cache/'+prefix+'temp_gs.pkl', 'wb') as f:
    pickle.dump(gs, f)
    f.close()

plt.scatter(gs[(y==0).flatten(),0],gs[(y==0).flatten(),1], label = 'ODE sample (0)', alpha = 0.2)
plt.scatter(gs[(y==1).flatten(),0],gs[(y==1).flatten(),1], label = 'ODE sample (1)', alpha = 0.2)
plt.legend();

# %%
plt.savefig('/work/zw122/tree_condsamp/LACE/tree-toy/cache/'+prefix+'temp_s.pdf')


