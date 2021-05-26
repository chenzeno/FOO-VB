from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import utils
import numpy as np
import time
import datasets as ds
import os


class Net100(nn.Module):
    def __init__(self):
        super(Net100, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Net200(nn.Module):
    def __init__(self):
        super(Net200, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train_continuous_mnist(args, model, device, train_loader, test_loader):
    ava_test = []
    weight_lst = utils.weight_lst(model)
    w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = \
        utils.init_param(weight_lst, args.s_init, device, True, args.alpha)
    for task in range(len(test_loader)):
        for epoch in range(1, args.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader[0]):
                model.train()
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 784)
                for mc_iter in range(args.train_mc_iters):
                    # Phi ~ MN(0,I,I)
                    phi_mat_lst = utils.gen_phi(w_mat_lst, device)
                    # W = M +B*Phi*A^t
                    utils.randomize_weights(weight_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                    output = model(data)
                    criterion = nn.CrossEntropyLoss()
                    loss = args.batch_size * criterion(output, target)
                    utils.zero_grad(weight_lst)
                    loss.backward()
                    grad_mat_lst = utils.weight_grad(weight_lst, device)
                    utils.aggregate_grads(args, avg_psi_mat_lst, grad_mat_lst)
                    utils.aggregate_e_a(args, e_a_mat_lst, grad_mat_lst, b_mat_lst, phi_mat_lst)
                    utils.aggregate_e_b(args, e_b_mat_lst, grad_mat_lst, a_mat_lst, phi_mat_lst)
                # M = M - B*B^t*avg_Phi*A*A^t
                utils.update_m(m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, args.eta)
                utils.update_a_b(a_mat_lst, b_mat_lst, e_a_mat_lst, e_b_mat_lst, device, args.use_gsvd)
                utils.zero_matrix(avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst)
            model.eval()
            with torch.no_grad():
                correct = 0
                for data, target in test_loader[task]:
                    data, target = data.to(device), target.to(device)
                    data = data.view(-1, 784)
                    for mc_iter in range(args.train_mc_iters):
                        phi_mat_lst = utils.gen_phi(w_mat_lst, device)
                        utils.randomize_weights(weight_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc = 100. * correct / (len(test_loader[task].dataset) * args.train_mc_iters)
            print('\nTask num {}, Epoch num {} Test Accuracy: {:.2f}%\n'.format(
                task, epoch, test_acc))
        test_acc_lst = []
        for i in range(task + 1):
            model.eval()
            with torch.no_grad():
                correct = 0
                for data, target in test_loader[i]:
                    data, target = data.to(device), target.to(device)
                    data = data.view(-1, 784)
                    for mc_iter in range(args.train_mc_iters):
                        phi_mat_lst = utils.gen_phi(w_mat_lst, device)
                        utils.randomize_weights(weight_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc = 100. * correct / (len(test_loader[i].dataset) * args.train_mc_iters)
                test_acc_lst.append(test_acc)
            print('\nTraning task Num: {} Test Accuracy of task {}: {:.2f}%\n'.format(
                task, i, test_acc))
        print(test_acc_lst)
        ava_test.append(np.average(np.asanyarray(test_acc_lst)))
    return ava_test


def train_multiple_tasks(args, model, device, train_loader, test_loader, perm_lst, save_path):
    ava_test = []
    weight_lst = utils.weight_lst(model)
    w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst = \
        utils.init_param(weight_lst, args.s_init, device, True, args.alpha)
    for task in range(len(perm_lst)):
        for epoch in range(1, args.epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                model.train()
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 784)
                data = data[:, perm_lst[task]]
                for mc_iter in range(args.train_mc_iters):
                    # Phi ~ MN(0,I,I)
                    phi_mat_lst = utils.gen_phi(w_mat_lst, device)
                    # W = M +B*Phi*A^t
                    utils.randomize_weights(weight_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                    output = model(data)
                    criterion = nn.CrossEntropyLoss()
                    loss = args.batch_size * criterion(output, target)
                    utils.zero_grad(weight_lst)
                    loss.backward()
                    grad_mat_lst = utils.weight_grad(weight_lst, device)
                    utils.aggregate_grads(args, avg_psi_mat_lst, grad_mat_lst)
                    utils.aggregate_e_a(args, e_a_mat_lst, grad_mat_lst, b_mat_lst, phi_mat_lst)
                    utils.aggregate_e_b(args, e_b_mat_lst, grad_mat_lst, a_mat_lst, phi_mat_lst)
                # M = M - B*B^t*avg_Phi*A*A^t
                utils.update_m(m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, args.eta)  # , task == 0)
                utils.update_a_b(a_mat_lst, b_mat_lst, e_a_mat_lst, e_b_mat_lst, device, args.use_gsvd)
                utils.zero_matrix(avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst)
            model.eval()
            with torch.no_grad():
                correct = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)
                    data = data.view(-1, 784)
                    data = data[:, perm_lst[task]]
                    for mc_iter in range(args.train_mc_iters):
                        phi_mat_lst = utils.gen_phi(w_mat_lst, device)
                        utils.randomize_weights(weight_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                train_acc = 100. * correct / (len(train_loader.dataset) * args.train_mc_iters)
                correct = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    data = data.view(-1, 784)
                    data = data[:, perm_lst[task]]
                    for mc_iter in range(args.train_mc_iters):
                        phi_mat_lst = utils.gen_phi(w_mat_lst, device)
                        utils.randomize_weights(weight_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc = 100. * correct / (len(test_loader.dataset) * args.train_mc_iters)
            print('\nTask num {}, Epoch num {}, Train Accuracy: {:.2f}% Test Accuracy: {:.2f}%\n'.format(
                task, epoch, train_acc, test_acc))
        test_acc_lst = []
        for i in range(task + 1):
            model.eval()
            with torch.no_grad():
                correct = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    data = data.view(-1, 784)
                    data = data[:, perm_lst[i]]
                    for mc_iter in range(args.train_mc_iters):
                        phi_mat_lst = utils.gen_phi(w_mat_lst, device)
                        utils.randomize_weights(weight_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst)
                        output = model(data)
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                test_acc = 100. * correct / (len(test_loader.dataset) * args.train_mc_iters)
                test_acc_lst.append(test_acc)
            print('\nTraning task Num: {} Test Accuracy of task {}: {:.2f}%\n'.format(
                task, i, test_acc))
        print(test_acc_lst)
        ava_test.append(np.average(np.asanyarray(test_acc_lst)))
    return ava_test


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of FOO-VB algorithm')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs per task (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--train_mc_iters', default=2500, type=int,
                        help='Number of MonteCarlo samples during training(default 10)')
    parser.add_argument('--s_init', default=0.27, type=float,
                        help='STD init value (default 0.27)')
    parser.add_argument('--eta', default=1, type=float,
                        help='STD init value (default 1)')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='STD init value (default 1)')
    parser.add_argument('--tasks', default=10, type=int,
                        help='number of tasks (default 10)')
    parser.add_argument('--results_dir', type=str, default="TMP",
                        help='Results dir name')
    parser.add_argument('--use_gsvd', action='store_true', help='use gsvd')
    parser.add_argument('--dataset', default="permuted_mnist", type=str, choices=['permuted_mnist', 'continuous_permuted_mnist'],
                        help='The name of the dataset to train. [Default: permuted_mnist]')
    parser.add_argument('--iterations_per_virtual_epc', default=468, type=int,
                        help='When using continuous dataset, number of iterations per epoch (in continuous mode, '
                             'epoch is not defined)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    utils.set_seed(args.seed)

    save_path = os.path.join("./logs", str(args.results_dir) + "/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if use_cuda else "cpu")
    if args.dataset == 'permuted_mnist':
        model = Net100().to(device)
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        perm_lst = utils.create_random_perm(args.tasks)
        ava_test = train_multiple_tasks(args, model, device, train_loader, test_loader, perm_lst, save_path)
        print(ava_test)
    else:
        if args.dataset == 'continuous_permuted_mnist':
            model = Net200().to(device)
            perm_lst = utils.create_random_perm(10)
            perm_lst = perm_lst[1:11]
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            train_loaders, test_loaders = ds.ds_padded_cont_permuted_mnist(num_epochs=int(args.epochs*args.tasks), iterations_per_virtual_epc=args.iterations_per_virtual_epc,
                                                                           contpermuted_beta=4, permutations=perm_lst,
                                                                           batch_size=args.batch_size, **kwargs)
            ava_test = train_continuous_mnist(args, model, device, train_loaders, test_loaders)
            print(ava_test)
    if args.save_model:
        torch.save(model.state_dict(), save_path + "/fcn.pt")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
