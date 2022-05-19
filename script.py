import math
import os
from pathlib import Path
from typing import Union, List
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# import wandb #TODO

random_seed = 8
np.random.seed(random_seed)
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
batch_size = 64
max_epochs = 1000
learning_rate = 0.05

how_many_test = 128*4
how_many_train = 1024*4
how_many_total = how_many_train + how_many_test

def gen_sythetic_data(num_points: int = how_many_total, num_features: int = 2, num_classes: int = 4, balance: Union[bool, List[int]] = True):
    data = np.zeros(shape=(num_points, num_features))
    classes = np.arange(num_classes)
    if balance is True:
        balance = np.ones(num_classes) * (1/num_classes)
    labels = np.random.choice(classes, size=num_points, p=balance)
    for ind, label in enumerate(labels):
        #TODO: Generalize > 4 classes
        if label == 0:
            data[ind] = np.random.multivariate_normal(mean=np.array([-0.5,0.25]), cov=np.array([[0.1,0.05], [0.05, 0.1]]), size=1)
        elif label == 1:
            data[ind] = np.random.laplace(loc=0.5, scale=0.1, size=num_features)
        elif label == 2:
            data[ind] = np.array([np.random.uniform(low=-0.2, high=0.6), np.random.uniform(low=0.0, high=-0.4)])
        else:
            data[ind] = np.array([np.random.beta(a=2, b=2)-0.5, np.random.beta(a=2, b=5)+0.5])
    return data, labels

class SyntheticData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def split_train_test(dataset, how_many_train, how_many_test):
    inds = np.arange(how_many_train + how_many_test)
    np.random.shuffle(inds)
    train_inds, test_inds = inds[how_many_test:], inds[:how_many_test]
    train_sampler = SubsetRandomSampler(train_inds)
    test_sampler = SubsetRandomSampler(test_inds)

    data_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    data_test = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return data_train, data_test

def plot(data: np.array, labels: np.array, x1: str, x2: str, y: str, title: str = '', show: bool = True, save: bool = False, figname='visualize_data.png', predictions: np.array = None):
    plt.figure(figsize=(14, 7))
    if predictions is not None:
        for x, y, l, p in zip(data[:,0], data[:,1], labels, predictions):
            c = "r" if l else "b"
            m = "+" if p else "o"
            plt.scatter(x=x, y=y, c=c, marker=m)
    else:
        for label in np.unique(labels):
            plt.scatter(x=data[labels == label][:,0], y=data[labels == label][:,1], label=f'y = {label}')
    plt.title(title, fontsize=20)
    plt.legend()
    if save:
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()


class NN(nn.Module):
    """Small neural network.
    Args:
        input_dimension (int): Number of dimensions in the input.
        hidden_dimension (int): Number of dimensions in the hidden layers.
        output_dimensiion (int): Number of dimensions in the output.
        activation (bool): Use activation.
        random_features (bool): If want to freeze randomly chosen first layer.
    """
    def __init__(self, input_dimension, hidden_dimension, output_dimension, activation=True, random_features=False):
        super(NN, self).__init__()

        self.activation = activation
        self.input_dimension = input_dimension
        self.in_layer = nn.Linear(input_dimension, hidden_dimension, bias = True)
        self.out_layer = nn.Linear(hidden_dimension, output_dimension)
        self.random_features = random_features

    def forward(self, x):
        x = self.in_layer(x)
        if self.random_featutres:
            x /= math.sqrt(self.input_dimension)
        if self.activation:
            x = F.relu(x)
        x = self.out_layer(x)
        return torch.sigmoid(x) #for now binary with 1 output neuron

def train_binary_classification(data_train_loader, hidden, num_epochs=max_epochs, learning_rate=learning_rate, random_features=False):
    model = NN(2, hidden, 1, True, random_features)
    model.to(device)
    if random_features:
        model.in_layer.weight.requires_grad = False
        model.in_layer.bias.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    criterion = nn.BCELoss() # nn.CrossEntropyLoss()
    losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for batch_idx, (x, y) in enumerate(data_train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)

            train_loss = criterion(y_hat, y.unsqueeze(1))
            train_loss.backward()

            y_pred_tag = torch.round(torch.sigmoid(y_hat))
            correct_results_sum = (y_pred_tag.view(-1,) == y).sum().float()
            acc = correct_results_sum/y.shape[0]
            acc = torch.round(acc * 100)

            running_loss += train_loss.item()
            running_acc += acc.item()
            optimizer.step()
        if epoch % 50 == 49:
            print(f"epoch {epoch}, loss {running_loss}, acc {running_acc/len(data_train_loader)}")
        losses.append(running_loss)
    return model, losses


def test_binary_classification(model, data_test_loader):
    model.eval()
    criterion = nn.BCELoss() # nn.CrossEntropyLoss()
    test_loss = 0.0
    test_acc = 0.0
    
    to_plot_x = np.zeros(shape=(len(data_test_loader), batch_size, 2)).astype(np.float32)
    to_plot_y = np.zeros(shape=(len(data_test_loader), batch_size)).astype(np.int8)
    to_plot_yhat = np.zeros(shape=(len(data_test_loader), batch_size)).astype(np.int8)

    for batch_idx, (x, y) in enumerate(data_test_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_pred_tag = torch.round(torch.sigmoid(y_hat))
        correct_results_sum = (y_pred_tag.view(-1,) == y).sum().float()
        batch_acc = correct_results_sum/y.shape[0]
        batch_acc = torch.round(batch_acc * 100)
        batch_loss = criterion(y_hat, y.unsqueeze(1))
        test_loss += batch_loss.item()
        test_acc += batch_acc.item()

        to_plot_x[batch_idx] = x.cpu().numpy()
        to_plot_y[batch_idx] = y.cpu().numpy()
        to_plot_yhat[batch_idx] = y_pred_tag.flatten().detach().cpu().numpy()

    to_plot = [to_plot_x.reshape(-1, 2), to_plot_y.reshape(-1,), to_plot_yhat.reshape(-1,)]
    return test_loss, test_acc/len(data_test_loader), to_plot
        

def train_all(train_loader, test_loader, every_n = 4, num_epochs=max_epochs, random_features = False):
    # range 1<->R+1 where R = 2*n so 4M + 1 = 2*n + 1 -> M = n/2 (max)
    n = how_many_train
    how_many_hiddens = np.arange(start=1, stop=n//2, step=every_n)
    test_loss = np.zeros_like(how_many_hiddens).astype(np.float32)
    for i, hidden in tqdm(enumerate(how_many_hiddens)):
        model, losses = train_binary_classification(train_loader, hidden, num_epochs=num_epochs, random_features=random_features)
        test_loss_val, test_acc_val, to_plot = test_binary_classification(model, test_loader)
        print(hidden, "TRAINING", losses[-1], "VALID", test_acc_val, test_loss_val)
        test_loss[i] = test_loss_val
        
        Path(f"models_more/{hidden}").mkdir(exist_ok=True, parents=True)
        plot(data=to_plot[0], labels=to_plot[1], x1='x1', x2='x2', y='y', title=f'Dataset with 2 classes; Model with {hidden} hidden neurons', save=True, figname=f'models_more/{hidden}/test.png', predictions=to_plot[2], show=False)
        torch.save(model.state_dict(), f"models_more/{hidden}/" + f"{hidden}_trained.pt")

    return how_many_hiddens, test_loss



def main():
    my_data2, my_labels2 = gen_sythetic_data(num_classes=2)
    plot(data=my_data2, labels=my_labels2, x1='x1', x2='x2', y='y', title='Dataset with 2 classes', save=True, figname='plots/data_more.png', show=False)

    train_data = SyntheticData(torch.FloatTensor(my_data2), torch.FloatTensor(my_labels2)) #LongTensor vs FloatTensor
    train_loader, test_loader = split_train_test(train_data, how_many_train, how_many_test)

    attempts, test_loss = train_all(train_loader, test_loader, every_n=4, num_epochs=max_epochs)
    print(test_loss)
    plt.scatter(attempts, test_loss)
    plt.title("Trained NNs. {max_epochs} epochs. Test loss function", fontsize=20)
    plt.legend()
    plt.savefig(f"plots/nn_experiment_more_scatter.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.plot(attempts, test_loss)
    plt.title(f"Trained NNs. {max_epochs} epochs. Test loss function", fontsize=20)
    plt.legend()
    plt.savefig("plots/nn_experiment_more_plot.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    #TODO: Random Features
    

if __name__ == "__main__":
    main()