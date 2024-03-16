# %%
import sys
sys.path.append('LACE/tree-toy')
sys.path.append('../../boostPM_py')
import sampling
import importlib
import torch
import boostPM_py
import matplotlib.pyplot as plt
import rpy2.robjects as robj
importlib.reload(sampling)
importlib.reload(boostPM_py)

# %%
prefix = "gaussian"

# %%
# prepare training data and pretrained f and g
torch.manual_seed(1)
n = 1000
d_g = 2
c = torch.cat((torch.zeros((n,1)),torch.ones((n,1))), dim = 0).double()
Xtrain = torch.zeros(2*n, d_g).double()
Xtrain[(c==1).flatten(),:] = torch.randn(n, d_g).double() @ torch.tensor([[1, 0.3], [0.3, 1]]).double()
Xtrain[(c==0).flatten(),:] = (torch.randn(n, d_g) + 8.).double() @ torch.tensor([[1, -0.3], [-0.3, 1]]).double()
Xtrain = Xtrain.double()
#-------generator---------
out = boostPM_py.boosting(Xtrain)
robj.r('saveRDS')(out, '/hpc/home/zw122/tree_condsamp/LACE/tree-toy/pretrained/' + prefix + 'out.rds')

# %%
plt.scatter(Xtrain[(c==0).flatten(),0],Xtrain[(c==0).flatten(),1], label = 0)
plt.scatter(Xtrain[(c==1).flatten(),0],Xtrain[(c==1).flatten(),1], label = 1)
plt.legend();

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
importlib.reload(sampling)
ccf = sampling.CCF(x_space="toy_i", latent_dim = d_g, n_classes = 2)
ccf.f = model
sample_q = sampling._sample_q_dict['sample_q_sgld']
ld_kwargs = {'every_n_plot': 100, 'save_path': True, 'latent_dim': d_g, 'sgld_lr': 0.09/2.,
                 'sgld_std': 0.3, 'n_steps': 11}


# %%
y = torch.cat((torch.ones(500), torch.zeros(500)), axis = 0).to(torch.long)
s = sample_q(ccf = ccf, y = y, device=torch.device('cpu'), **ld_kwargs)

# %%
gs = ccf.g(s[1]).detach()
plt.scatter(gs[(y==0).flatten(),0],gs[(y==0).flatten(),1], label = 0)
plt.scatter(gs[(y==1).flatten(),0],gs[(y==1).flatten(),1], label = 1)
plt.legend();

# %%
for _ in range(len(s[0])):
    gs = ccf.g(s[0][_]).detach()
    plt.figure()
    plt.scatter(gs[(y==0).flatten(),0],gs[(y==0).flatten(),1], label = 0)
    plt.scatter(gs[(y==1).flatten(),0],gs[(y==1).flatten(),1], label = 1)
    plt.legend();

# %%
import pickle
with open('/hpc/home/zw122/tree_condsamp/LACE/tree-toy/cache/'+prefix+'temp_s.pkl', 'wb') as f:
    pickle.dump(s, f)
    f.close()

# %%


# %%


# %%



