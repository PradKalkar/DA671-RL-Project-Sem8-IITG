# Group 12: Neural TS
# Pradnesh Prasad Kalkar: 190101103
# Anirudh Phukan: 190101104
# Saket Kumar Singh: 190101081
# Abhishek Agrahari: 190123066
# Akshat Agrawal: 190123069
# Manish Kumar: 190123067

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_openml
import sklearn.utils
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from backpack import backpack, extend
from backpack.extensions import BatchGrad
import argparse

# Main function
def main():
    # reading command line arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    dataset = args.dataset
    llambda = args.llambda
    nu = args.nu
    hidden = args.hidden
    num_rounds_threshold = args.num_rounds_threshold

    mab_obj = MAB_dataset(dataset)
    model = NeuralTS(mab_obj.context_dim, llambda, nu, hidden)

    regret_list = []
    for t in range(min(num_rounds_threshold, mab_obj.num_samples)):
        arm_context_matrix, reward_vector = mab_obj.step() # obtained from the dataset (GT)
        chosen_arm = model.select(arm_context_matrix)
        r = reward_vector[chosen_arm] # inspect the GT reward for chosen arm
        regret = np.max(reward_vector) - r # regret is max GT reward - GT reward for the chosen arm
        model.train(arm_context_matrix[chosen_arm], r)
        regret_list.append(regret)

        if t % 100 == 0:
            print('Round {}: Total regret: {:.3f}, Chosen Arm: {}'.format(t, np.sum(regret_list), chosen_arm))


############################# HELPER FUNCTIONS ####################################

def get_args_parser():
    parser = argparse.ArgumentParser(description='Thompson Sampling')
    parser.add_argument('--dataset', default='mushroom', metavar='DATASET', help='name of the dataset') 
    parser.add_argument('--llambda', default=0.01, type=float, help='regularization hyperparameter')
    parser.add_argument('--nu', default=0.001, type=float, help='exploration variance') 
    parser.add_argument('--hidden', default=100, type=int, help='number of neurons in each hidden layer')
    parser.add_argument('--num_rounds_threshold', default=10000, type=int, help='Upper limit for the number of rounds')
    return parser

# a class for representing MAB_dataset
class MAB_dataset(object):
    def __init__(self, dataset_name, is_shuffle=True, random_state=42):

        # Pull data using fetch_openml
        if dataset_name == 'mushroom':
            X, y = fetch_openml('mushroom', version=1, return_X_y=True) # X.shape is (8124, 22)
        elif dataset_name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True) # X.shape is (19020, 10)
        elif dataset_name == 'adult':
            X, y = fetch_openml('adult', version=2, return_X_y=True) # X.shape is (48842, 14)
        elif dataset_name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True) # X.shape is (58000, 9)
        elif dataset_name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True) # X.shape is (70000, 784)
        elif dataset_name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True) # X.shape is (581012, 54)
        
        X[np.isnan(X)] = -1 # replace the NaN values by -1
        X = normalize(X)

        # Shuffle data
        if is_shuffle:
            self.X, self.y = sklearn.utils.shuffle(X, y, random_state=random_state)
        else:
            self.X, self.y = X, y

        # y_arm will just assign discrete class numbers to the classes in y -> basically a classification label
        self.y_arm = OrdinalEncoder(dtype=np.int).fit_transform(self.y.reshape((-1, 1))) # y_arm.shape -> (num_examples, 1)

        # some helper variables below
        self.idx = 0
        self.num_samples = self.y.shape[0]
        self.num_arms = np.max(self.y_arm) + 1 # num_arms is basically the number of classification classes in y
        self.feature_dim = self.X.shape[1]
        self.context_dim = self.feature_dim * self.num_arms # dimension of the context vector


    def step(self):
        assert (self.idx < self.num_samples)

        X = np.zeros((self.num_arms, self.context_dim))
        for a in range(self.num_arms):
            # for all arms a, set the corresponding col entries to the same context vector located at idx in X
            X[a, a*self.feature_dim : a*self.feature_dim+self.feature_dim] = self.X[self.idx] # rest all cols will be 0 for row a

        arm = self.y_arm[self.idx][0] # arm is basically the class label

        rewards = np.zeros((self.num_arms,)) # rewards is a one-hot vector with rewards[i] = 1 only for label arm
        rewards[arm] = 1

        self.idx += 1
        return X, rewards


    def finish(self):
        return self.idx == self.num_samples


    def reset(self):
        self.idx = 0


class Mean_Estimator(nn.Module):
    """
        Our Neural Network of 2 layers. This is used to estimate the mean of the posterior distribution.
    """
    def __init__(self, context_dim, hidden_size):
        super(Mean_Estimator, self).__init__()
        self.fc1 = nn.Linear(context_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.fc1.weight.data.normal_(0.0, 4 / hidden_size)
        self.fc2.weight.data.normal_(0.0, 2 / hidden_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NeuralTS(object):
    def __init__(self, context_dim, llambda=1, nu=1, hidden=100):
        self.mean_estimator = extend(Mean_Estimator(context_dim, hidden_size=hidden).cuda()) # extend enables to use functionalities of backpack for our NN
        self.context_list = None
        self.len = 0
        self.reward = None
        self.llambda = llambda
        self.nu = nu
        self.mse_loss = nn.MSELoss()
        self.total_param = sum(p.numel() for p in self.mean_estimator.parameters() if p.requires_grad)
        self.U = llambda * torch.ones((self.total_param,)).cuda()

    def select(self, arm_context_matrix):
        tensor = torch.from_numpy(arm_context_matrix).float().cuda() # moving arm_context_matrix to GPU
        mu = self.mean_estimator(tensor) # estimate the mean for all the arms
        
        sum_mu = torch.sum(mu)
        with backpack(BatchGrad()):
            sum_mu.backward()

        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.mean_estimator.parameters()], dim=1)

        sigma = torch.sqrt(torch.sum(self.llambda * self.nu * self.nu * g_list * g_list / self.U, dim=1))
        sampled_rewards = torch.normal(mu.view(-1), sigma.view(-1)) # sample reward for each arm using gaussian
        chosen_arm = torch.argmax(sampled_rewards)

        self.U += g_list[chosen_arm] * g_list[chosen_arm]
        return chosen_arm
    
    def train(self, arm_context_vector, reward):
        self.len += 1
        optimizer = optim.SGD(self.mean_estimator.parameters(), lr=1e-2, weight_decay=self.llambda / self.len)
        if self.context_list is None:
            self.context_list = torch.from_numpy(arm_context_vector.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.reward = torch.tensor([reward], device='cuda', dtype=torch.float32) # GT reward for the chosen arm
        else:
            # append new context vector and reward to prev
            self.context_list = torch.cat((self.context_list, torch.from_numpy(arm_context_vector.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.reward = torch.cat((self.reward, torch.tensor([reward], device='cuda', dtype=torch.float32)))

        for _ in range(100): # perform gradient descent for 100 steps 
            self.mean_estimator.zero_grad()
            optimizer.zero_grad()
            pred_mean = self.mean_estimator(self.context_list).view(-1) # run the forward pass on all the context vectors seen till now
            loss = self.mse_loss(pred_mean, self.reward)
            loss.backward()
            optimizer.step()


# entry point
if __name__ == '__main__':
    main()