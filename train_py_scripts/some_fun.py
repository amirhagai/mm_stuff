
from __future__ import print_function
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torchvision

import os
from autoattack import AutoAttack

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict
import torch.nn as nn
import math
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST, MNIST

import time


#@title TRADES loss

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss



#@title CNN definition

class SmallCNN(nn.Module):
    def __init__(self, drop=0.5, num_labels=10):
        super(SmallCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = num_labels

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits
    

#@title data handling

class BinaryMNIST(Dataset):
    """A custom Torch Dataset to handle filtering and relabeling MNIST for binary classification."""
    def __init__(self, mnist_dataset, class1, class2):
        """
        mnist_dataset: Original MNIST dataset.
        class1: First class (will be labeled as 0).
        class2: Second class (will be labeled as 1).
        """
        # Filter and remap classes
        self.data = []
        self.targets = []

        for img, label in mnist_dataset:
            if label == class1:
                self.data.append(img)
                self.targets.append(0)
            elif label == class2:
                self.data.append(img)
                self.targets.append(1)

        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)
        print(len(self.data))
        print(len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def create_data_loaders(batch_size_train, batch_size_test, class1, class2, data_dir='../data'):
    transform = transforms.ToTensor()
    if args.data == "MNIST":
        full_train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        full_test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    else:
        full_train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        full_test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    train_dataset = BinaryMNIST(full_train_dataset, class1, class2)
    test_dataset = BinaryMNIST(full_test_dataset, class1, class2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    return train_loader, test_loader



def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_binary_mnist(
    n_examples: Optional[int] = None,
    data_dir: str = '../data',
    train: bool = False,
    class1:int = 0,
    class2:int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.ToTensor()
    if args.data == "MNIST":
        dataset = datasets.MNIST(root=data_dir,
                                train=train,
                                transform=transform,
                                download=True)
    else:
        dataset = datasets.FashionMNIST(root=data_dir,
                                train=train,
                                transform=transform,
                                download=True)        
    dataset = BinaryMNIST(dataset, class1, class2)
    return _load_dataset(dataset, n_examples)

# x_test_tensor, y_test_tensor = load_binary_mnist(n_examples=50, class1=1, class2=7, train=False)
# len(y_test_tensor)



#@title Robust evaluation


def pgd_whitebox(model,
                  X,
                  y,
                  epsilon=8./255,
                  num_steps=400,
                  step_size=0.04,
                  norm="inf"):

    print("start normal pgd", end="\n\n")
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    start = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    if norm == "2":
        random_noise = random_noise.renorm_(p=2, dim=0, maxnorm=epsilon / 4)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    x_copy = X.clone().detach()

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()

        if norm == "inf":
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        elif norm == "2":
            eta = step_size * X_pgd.grad.data / torch.norm(X_pgd.grad.data)
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = (X_pgd.data - X.data).renorm_(p=2, dim=0, maxnorm=epsilon)

        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print(f'\tstart err - {start} out of {len(y)}')
    print(f'\tpgd err - {err_pgd} out of {len(y)}')
    return err, err_pgd


def get_correct_class_logits(model, x, y):
    logits = model(x)
    correct_class_logits = logits.gather(1, y.view(-1, 1)).squeeze()
    return correct_class_logits

def eval_robust(model, n_examples=50, class1=1, class2=7, epsilon=8./255, sample=False):
    x_test, y_test = load_binary_mnist(n_examples=n_examples, class1=class1, class2=class2, train=False)

    x_train, y_train = load_binary_mnist(n_examples=n_examples, class1=class1, class2=class2, train=True)

    l2_eps =  math.sqrt(.1**2 * 28 * 28)

    print("\n\n run train evaluation l inf \n\n")

    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    x_adv_train_linf = adversary.run_standard_evaluation(x_train, y_train)

    print("\n\n run train evaluation l 2 \n\n")
    adversary = AutoAttack(model, norm='L2', eps=l2_eps, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    x_adv_train_l2 = adversary.run_standard_evaluation(x_train, y_train)


    print("\n\n run test evaluation l inf \n\n")

    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    x_adv_test_linf = adversary.run_standard_evaluation(x_test, y_test)

    print("\n\n run test evaluation l 2 \n\n")
    adversary = AutoAttack(model, norm='L2', eps=l2_eps, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    x_adv_test_l2 = adversary.run_standard_evaluation(x_test, y_test)   

    return x_adv_train_linf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2, x_test, y_test, x_train, y_train



#@title utils

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 55:
        lr = args.lr * 0.1
    if epoch >= 75:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#@title train functions

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, target)


        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def train_trades(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # outputs = model(data)
        # loss = criterion(outputs, target)

        loss = trades_loss(model=model,
                    x_natural=data,
                    y=target,
                    optimizer=optimizer,
                    step_size=args.step_size,
                    epsilon=args.epsilon,
                    perturb_steps=args.num_steps,
                    beta=args.beta)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))








def imshow(img, name, c1, c2, show_fig=False):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xticks([], [])
    plt.yticks([], [])
    os.makedirs(f"/data/imgs_{args.data}/{args.class1}_{args.class2}", exist_ok=True)
    plt.savefig(f"/data/imgs_{args.data}/{args.class1}_{args.class2}/{name}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    if show_fig:
      plt.show()



#@title main

class Config:
    def __init__(self, batch_size=128, test_batch_size=128, epochs=100, lr=0.01,
                 momentum=0.9, no_cuda=False, epsilon=0.3, num_steps=40,
                 step_size=0.01, beta=1.0, seed=1, log_interval=100,
                 model_dir='./model-mnist-smallCNN', save_freq=5, class1=1, class2=7, trades=True, data="MNIST"):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.no_cuda = no_cuda
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.beta = beta
        self.seed = seed
        self.log_interval = log_interval
        self.model_dir = model_dir
        self.save_freq = save_freq
        self.class1 = class1
        self.class2 = class2
        self.trades = trades
        self.data = data




def main(model, args, optimizer, device, class1, class2, trades=True):
    # init model, Net() can be also used here for training
    train_loader, test_loader = create_data_loaders(args.batch_size, args.test_batch_size, class1, class2)
    start = time.time()
    if trades:
      print("train using trades")
    else:
      print("train using normal CE loss")
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        if trades:
        # adversarial training
          train_trades(args, model, device, train_loader, optimizer, epoch)
        else:
          train(args, model, device, train_loader, optimizer, epoch)
        if epoch % 10 == 0 and epoch > 0 :
          # evaluation on natural examples
          print('================================================================')
          eval_train(model, device, train_loader)
          eval_test(model, device, test_loader)
          print('================================================================')

    # x_adv_train_inf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2, x_train, x_test = eval_robust(model, n_examples=2, class1=class1, class2=class2, epsilon=args.epsilon)
    end = time.time()
    print(f"time taken - {end - start}")
    return model #, x_adv_train_inf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2, x_train, x_test





train_with_trades= True #@param {type:"boolean"}
class_1= 8 #@param {type:"integer"}
class_2= 7 #@param {type:"integer"}
epochs= 10 #@param {type:"integer"}
num_grad_steps_trades_inner_maximization = 10 #@param {type:"integer"}
trades_epsilon = 0.3 #@param {type:"number"}
data_ = "MNIST"

n_examples=50




for data_ in ["MNIST", "FashionMNIST"]:

    rob_acc_linf_train_arr = []
    rob_acc_l2_train_arr = []
    rob_acc_linf_test_arr = []
    rob_acc_l2_test_arr = []
    acc_train_arr = []
    acc_test_arr = []
    for i in range(0, 10, 1):
        for j in range(i+1, 10, 1):
            print(f"train over classes {i} and {j}", end="\n\n\n\n")
            args = Config(epochs=epochs, epsilon=trades_epsilon , num_steps=num_grad_steps_trades_inner_maximization, class1=i, class2=j, trades=train_with_trades, data=data_)
            # settings
            model_dir = args.model_dir
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            use_cuda = not args.no_cuda and torch.cuda.is_available()
            torch.manual_seed(args.seed)
            device = torch.device("cuda" if use_cuda else "cpu")
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            model = SmallCNN(num_labels=2).to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

            model = main(model, args, optimizer, device, args.class1, args.class2, args.trades)

            x_adv_train_linf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2, x_test, y_test, x_train, y_train = \
                eval_robust(model, n_examples=n_examples, class1=i, class2=j, epsilon=trades_epsilon, sample=False)
            
            torch.cuda.synchronize()

            with torch.no_grad():

                x_adv_train_linf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2, x_test, y_test, x_train, y_train =\
                x_adv_train_linf.to("cuda"), x_adv_train_l2.to("cuda"), x_adv_test_linf.to("cuda"), x_adv_test_l2.to("cuda"), x_test.to("cuda"), y_test.to("cuda"), x_train.to("cuda"), y_train.to("cuda")

                ind_train_linf = (model(x_adv_train_linf).argmax(dim=1) != y_train) * (model(x_train).argmax(dim=1) == y_train )
                ind_train_l2 = (model(x_adv_train_l2).argmax(dim=1) != y_train) * (model(x_train).argmax(dim=1) == y_train )
                ind_test_linf = (model(x_adv_test_linf).argmax(dim=1) != y_test) * (model(x_test).argmax(dim=1) == y_test )
                ind_test_l2 = (model(x_adv_test_l2).argmax(dim=1) != y_test) * (model(x_test).argmax(dim=1) == y_test )


                rob_acc_linf_train = (model(x_adv_train_linf).argmax(dim=1) != y_train).sum() / len(y_train)
                rob_acc_l2_train = (model(x_adv_train_l2).argmax(dim=1) != y_train).sum() / len(y_train)
                rob_acc_linf_test = (model(x_adv_test_linf).argmax(dim=1) != y_test).sum() / len(y_test)
                rob_acc_l2_test = (model(x_adv_test_l2).argmax(dim=1) != y_test).sum() / len(y_test)

                acc_train = (model(x_train).argmax(dim=1) == y_train ).sum() / len(y_train)
                acc_test = (model(x_test).argmax(dim=1) == y_test ).sum() / len(y_test)


                rob_acc_linf_train_arr.append(rob_acc_linf_train.item())
                rob_acc_l2_train_arr.append(rob_acc_l2_train.item())
                rob_acc_linf_test_arr.append(rob_acc_linf_test.item())
                rob_acc_l2_test_arr.append(rob_acc_l2_test.item())
                acc_train_arr.append(acc_train.item())
                acc_test_arr.append(acc_test.item())

                x_adv_train_linf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2 =\
                    x_adv_train_linf[ind_train_linf], x_adv_train_l2[ind_train_l2], x_adv_test_linf[ind_test_linf], x_adv_test_l2[ind_test_l2]
                
                x_train_linf, y_train_linf = x_train[ind_train_linf], y_train[ind_train_linf]
                x_train_l2, y_train_l2 = x_train[ind_train_l2], y_train[ind_train_l2]
                x_test_linf, y_test_linf = x_test[ind_test_linf], y_test[ind_test_linf]
                x_test_l2, y_test_l2 = x_test[ind_test_l2], y_test[ind_test_l2]

                x_train_linf = torch.cat([x_train_linf, x_adv_train_linf]).detach().cpu().repeat(1, 3, 1, 1)
                x_train_l2 = torch.cat([x_train_l2, x_adv_train_l2]).detach().cpu().repeat(1, 3, 1, 1)
                x_test_linf = torch.cat([x_test_linf, x_adv_test_linf]).detach().cpu().repeat(1, 3, 1, 1)
                x_test_l2 = torch.cat([x_test_l2, x_adv_test_l2]).detach().cpu().repeat(1, 3, 1, 1)

                if len(x_train_linf) > 0:
                    imshow(torchvision.utils.make_grid(x_train_linf, nrow=int(len(x_train_linf) / 2)), "x_train_linf", i, j)
                    for k in range(int(len(x_train_linf) / 2)):
                        imshow(torchvision.utils.make_grid(torch.cat([x_train_linf[k][None,...,], x_train_linf[int(len(x_train_linf) / 2) + k][None,...,]]), nrow=2), f"x_train_linf_{k}", i, j)

                if len(x_train_l2) > 0:
                    imshow(torchvision.utils.make_grid(x_train_l2, nrow=int(len(x_train_l2) / 2)), "x_train_l2", i, j)
                    for k in range(int(len(x_train_l2) / 2)):
                        imshow(torchvision.utils.make_grid(torch.cat([x_train_l2[k][None,...,], x_train_l2[int(len(x_train_l2) / 2) + k][None,...,]]), nrow=2), f"x_train_l2_{k}", i, j)

                if len(x_test_linf) > 0:
                    imshow(torchvision.utils.make_grid(x_test_linf, nrow=int(len(x_test_linf) / 2)), "x_test_linf", i, j)
                    for k in range(int(len(x_test_linf) / 2)):
                        imshow(torchvision.utils.make_grid(torch.cat([x_test_linf[k][None,...,], x_test_linf[int(len(x_test_linf) / 2) + k][None,...,]]), nrow=2), f"x_test_linf_{k}", i, j)

                if len(x_test_l2) > 0:
                    imshow(torchvision.utils.make_grid(x_test_l2, nrow=int(len(x_test_l2) / 2)), "x_test_l2", i, j)
                    for k in range(int(len(x_test_l2) / 2)):
                        imshow(torchvision.utils.make_grid(torch.cat([x_test_l2[k][None,...,], x_test_l2[int(len(x_test_l2) / 2) + k][None,...,]]), nrow=2), f"x_test_l2_{k}", i, j)


        rob_acc_linf_train_arr = np.array(rob_acc_linf_train_arr)
        rob_acc_l2_train_arr = np.array(rob_acc_l2_train_arr)
        rob_acc_linf_test_arr = np.array(rob_acc_linf_test_arr )
        rob_acc_l2_test_arr = np.array(rob_acc_l2_test_arr )
        acc_train_arr = np.array(acc_train_arr)
        acc_test_arr = np.array(acc_test_arr)



        print(f"robust accuracy linf train mean - {rob_acc_linf_train_arr.mean()}, std - {rob_acc_linf_train_arr.std()}")
        print(f"robust accuracy l2 train mean - {rob_acc_l2_train_arr.mean()}, std - {rob_acc_l2_train_arr.std()}")
        print(f"robust accuracy linf test mean - {rob_acc_linf_test_arr.mean()}, std - {rob_acc_linf_test_arr.std()}")
        print(f"robust accuracy l2 test mean - {rob_acc_l2_test_arr.mean()}, std - {rob_acc_l2_test_arr.std()}")


        print(f"accuracy train - {acc_train_arr.mean()}, std - {acc_train_arr.std()}")
        print(f"accuracy test mean - {acc_test_arr.mean()}, std - {acc_test_arr.std()}")

    # path = f"/content/drive/MyDrive/thesis/adv_examples_mnist_fasion/{args.class1}_{args.class2}"
    # os.makedirs(path, exist_ok=True)
    # os.makedirs(f"/content/drive/MyDrive/Colab Notebooks/THS/{args.class1}_{args.class2}", exist_ok=True)
    # torch.save(model.state_dict(), f"/content/drive/MyDrive/Colab Notebooks/THS/{args.class1}_{args.class2}/mdl.pth")
    # x_adv_train_inf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2, x_train_linf, x_train_l2, x_test_linf, x_test_l2 = eval_robust(model, 20, args.class1, args.class2, args.epsilon, sample=True)
    # x_adv_train_inf, x_adv_train_l2, x_adv_test_linf, x_adv_test_l2, x_train_linf, x_train_l2, x_test_linf, x_test_l2 = x_adv_train_inf.cpu().detach(), x_adv_train_l2.cpu().detach(), x_adv_test_linf.cpu().detach(), x_adv_test_l2.cpu().detach(), x_train_linf.cpu().detach(), x_train_l2.cpu().detach(), x_test_linf.cpu().detach(), x_test_l2.cpu().detach()


    # imshow(torchvision.utils.make_grid(torch.cat([x_test_linf, x_adv_test_linf]), nrow=5), f"test_linf", args.class1, args.class2)
    # imshow(torchvision.utils.make_grid(torch.cat([x_test_l2, x_adv_test_l2]), nrow=5), f"test_l2", args.class1, args.class2)
    # imshow(torchvision.utils.make_grid(torch.cat([x_train_linf, x_adv_train_inf]), nrow=5), f"train_linf", args.class1, args.class2)
    # imshow(torchvision.utils.make_grid(torch.cat([x_train_l2, x_adv_train_l2]), nrow=5), f"train_l2", args.class1, args.class2)