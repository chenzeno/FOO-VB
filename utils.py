import torch
import math
import numpy as np
import random


def create_random_perm(n_permutations):
    """
        This function returns a list of array permutation (size of 28*28 = 784) to create permuted MNIST data.
        Note the first permutation is the identity permutation.
        :param n_permutations: number of permutations.
        :return perm_lst: a list of permutations.
    """
    perm_lst = [np.arange(784)]
    for seed in range(1, n_permutations):
        np.random.seed(seed)
        perm_lst.append(np.random.permutation(784))

    return perm_lst


def solve_matrix_equation(v_mat, e_mat, print_norm_flag=False):
    """
        This function returns a solution for the following non-linear matrix equation XX^{\top}+VEX^{\top}-V = 0.
        All the calculations are done in double precision.
        :param v_mat: N*N PD matrix.
        :param e_mat: N*N matrix.
        :param print_norm_flag: Boolean parameter. Print the norm of the matrix equation.
        :return: x_mat: N*N matrix a solution to the non-linear matrix equation.
    """
    # B = V + (1/4)V*E*(E^T)*V
    v_mat = v_mat.double()
    e_mat = e_mat.double()
    ve_product = torch.mm(v_mat, e_mat)
    b_mat = torch.add(v_mat, 0.25, torch.mm(ve_product, torch.transpose(ve_product, 0, 1)))
    left_mat, diag_mat, right_mat = torch.svd(b_mat)

    assert (torch.min(diag_mat).item() > 0), "v_mat is singular!"

    # L = B^{1/2}
    l_mat = torch.mm(torch.mm(left_mat, torch.diagflat(torch.sqrt(diag_mat))),
                     torch.transpose(right_mat, 0, 1))
    inv_l_mat = torch.mm(torch.mm(right_mat, torch.diagflat(torch.reciprocal(torch.sqrt(diag_mat)))),
                         torch.transpose(left_mat, 0, 1))
    # L^-1*V*E=S*Lambda*W^t (SVD)
    s_mat, lambda_mat, w_mat = torch.svd(torch.mm(inv_l_mat, ve_product))
    # Q = S*W^t
    q_mat = torch.mm(s_mat, torch.transpose(w_mat, 0, 1))
    # X = L*Q-(1/2)V*E
    x_mat = torch.add(torch.mm(l_mat, q_mat), -1 / 2, ve_product)
    if print_norm_flag:
        mat = torch.add(torch.add(torch.mm(x_mat, torch.transpose(x_mat, 0, 1)),
                                  torch.mm(ve_product, torch.transpose(x_mat, 0, 1))), -1, v_mat)
        mat_norm = torch.norm(mat)
        print('The Frobenius norm of the matrix is', mat_norm.item())
    return x_mat.float()


def lst_to_device(device, tensor_lst):
    """
        :param device: device index to select.
        :param tensor_lst: list of tensors
        :return:
    """
    for idx, i in enumerate(tensor_lst):
        tensor_lst[idx] = tensor_lst[idx].to(device)


def weight_lst(self):
    """
        :param self.
        :return: A list of iterators of the network parameters.
    """
    return [w for w in self.parameters()]


def weight_grad(tensor_lst, device):
    """
        This function return a list of matrices containing the gradient of the network parameters for each layer.
        :param tensor_lst: A list of iterators of the network parameters.
        :param device: device index to select.
        :return: grad_mat_lst: A list of matrices containing the gradients of the network parameters.
    """
    grad_mat_lst = []
    for i in range(len(tensor_lst)):
        if i % 2:
            continue
        else:
            grad_mat_lst.append(torch.cat((tensor_lst[i].grad.data,
                                           torch.unsqueeze(tensor_lst[i + 1].grad.data, 1)), 1))
    lst_to_device(device, grad_mat_lst)
    return grad_mat_lst


def init_param(tensor_lst, s_init, device, use_custom_init = False, alpha = 0.5):
    """
        :param tensor_lst: A list of iterators of the network parameters.
        :param s_init: Init value of the diagonal of a and b.
        :param device: device index to select.
        :return: w_mat_lst: A list of matrices in size of P*N.
        :return: m_mat_lst: A list of matrices in size of P*N.
        :return: a_mat_lst: A list of matrices in size of N*N.
        :return: b_mat_lst: A list of matrices in size of P*P.
        :return: avg_psi_mat_lst: A list of matrices in size of P*N.
        :return: e_a_mat_lst: A list of matrices in size of N*N.
        :return: e_b_mat_lst: A list of matrices in size of P*P.
    """
    w_mat_lst = []
    m_mat_lst = []
    a_mat_lst = []
    b_mat_lst = []
    avg_psi_mat_lst = []
    e_a_mat_lst = []
    e_b_mat_lst = []
    for i in range(len(tensor_lst)):
        if i % 2:
            continue
        else:
            w_mat_lst.append(torch.zeros(tensor_lst[i].size()[0], tensor_lst[i].size()[1] + 1))
            avg_psi_mat_lst.append(torch.zeros(tensor_lst[i].size()[0], tensor_lst[i].size()[1] + 1))
            if use_custom_init:
                m_mat_lst.append(math.sqrt((2.0*alpha/(tensor_lst[i].size()[1] + 2.0))) * torch.randn_like(
                    torch.cat((tensor_lst[i].data, torch.unsqueeze(tensor_lst[i + 1], 1)), 1), device=device))
                a_mat_lst.append(torch.diagflat(math.sqrt(math.sqrt((2.0*(1.0-alpha)/(tensor_lst[i].size()[1] + 2.0)))) * torch.ones(tensor_lst[i].size()[1] + 1)))
                b_mat_lst.append(torch.diagflat(math.sqrt(math.sqrt((2.0*(1.0-alpha)/(tensor_lst[i].size()[1] + 2.0)))) * torch.ones(tensor_lst[i].size()[0])))
            else:
                # m_mat_lst.append(torch.cat((tensor_lst[i].data, torch.unsqueeze(tensor_lst[i + 1].data, 1)), 1))
                m_mat_lst.append(torch.cat((math.sqrt(2.0/(tensor_lst[i].size()[0] + tensor_lst[i].size()[1])) *
                                           torch.randn_like(tensor_lst[i].data, device=device),
                                           math.sqrt(2.0/(1.0 + tensor_lst[i].size()[1])) *
                                           torch.randn_like(torch.unsqueeze(tensor_lst[i + 1].data, 1))), 1))
                a_mat_lst.append(torch.diagflat(s_init * torch.ones(tensor_lst[i].size()[1]+1)))
                b_mat_lst.append(torch.diagflat(s_init * torch.ones(tensor_lst[i].size()[0])))
            e_a_mat_lst.append(torch.zeros(tensor_lst[i].size()[1] + 1, tensor_lst[i].size()[1]+1))
            e_b_mat_lst.append(torch.zeros(tensor_lst[i].size()[0], tensor_lst[i].size()[0]))
    lst_to_device(device, w_mat_lst)
    lst_to_device(device, avg_psi_mat_lst)
    lst_to_device(device, m_mat_lst)
    lst_to_device(device, a_mat_lst)
    lst_to_device(device, b_mat_lst)
    lst_to_device(device, e_a_mat_lst)
    lst_to_device(device, e_b_mat_lst)
    return w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst


def update_weight(tensor_lst, w_mat_lst):
    """
        This function update the parameters of the network.
        :param tensor_lst: A list of iterators of the network parameters.
        :param w_mat_lst: A list of matrices in size of P*N.
        :return:
    """
    for i in range(len(tensor_lst)):
        if i % 2:
            continue
        else:
            tensor_lst[i].data.copy_(w_mat_lst[int(i/2)][:, :-1])
            tensor_lst[i + 1].data.copy_(w_mat_lst[int(i/2)][:, -1])


def gen_phi(w_mat_lst, device):
    """
        :param w_mat_lst: A list of matrices in size of P*N.
        :param device: device index to select.
        :return phi_mat_lst: A list of normal random matrices in size of P*N.
    """
    phi_mat_lst = []
    for i in w_mat_lst:
        phi_mat_lst.append(torch.randn_like(i, device=device))
    lst_to_device(device, phi_mat_lst)
    return phi_mat_lst


def randomize_weights(tensor_lst, w_mat_lst, m_mat_lst, a_mat_lst, b_mat_lst, phi_mat_lst):
    """
        This function generate a sample of normal random weights with mean M and covariance matrix of (A*A^t)\otimes(B*B^t)
        (\otimes = kronecker product). In matrix form the update rule is W = M + B*Phi*A^t.
        :param tensor_lst: A list of iterators of the network parameters.
        :param w_mat_lst: m_mat_lst: A list of matrices in size of P*N.
        :param m_mat_lst: m_mat_lst: A list of matrices in size of P*N.
        :param a_mat_lst: A list of matrices in size of N*N.
        :param b_mat_lst: A list of matrices in size of P*P.
        :param phi_mat_lst: A list of normal random matrices in size of P*N.
        :return:
    """
    for i in range(len(w_mat_lst)):
        # W = M + B*Phi*A^t
        w_mat_lst[i].copy_(torch.add(m_mat_lst[i], torch.mm(torch.mm(b_mat_lst[i], phi_mat_lst[i]),
                                                            torch.transpose(a_mat_lst[i], 0, 1))))
    update_weight(tensor_lst, w_mat_lst)


def zero_grad(tensor_lst):
    """
        :param tensor_lst: A list of iterators of the network parameters.
        :return:
    """
    for i in tensor_lst:
        if i.grad is not None:
            i.grad.detach_()
            i.grad.zero_()


def zero_matrix(avg_psi_mat_lst, e_a_mat_lst, e_b_mat_lst):
    """
        :param avg_psi_mat_lst: A list of matrices in size of P*N.
        :param e_a_mat_lst: A list of matrices in size of N*N.
        :param e_b_mat_lst: A list of matrices in size of P*P.
        :return:
    """
    for i in range(len(avg_psi_mat_lst)):
        avg_psi_mat_lst[i].copy_(torch.zeros_like(avg_psi_mat_lst[i]))
    for i in range(len(e_a_mat_lst)):
        e_a_mat_lst[i].copy_(torch.zeros_like(e_a_mat_lst[i]))
    for i in range(len(e_b_mat_lst)):
        e_b_mat_lst[i].copy_(torch.zeros_like(e_b_mat_lst[i]))


def aggregate_grads(args, avg_psi_mat_lst, grad_mat_list):
    """
        This function estimate the expectation of the gradient using Monte Carlo average.
        :param args: Training settings.
        :param avg_psi_mat_lst: A list of matrices in size of P*N.
        :param grad_mat_list: A list of matrices in size of P*N.
        :return:
    """
    for i in range(len(grad_mat_list)):
        avg_psi_mat_lst[i].add_((1/args.train_mc_iters)*grad_mat_list[i])


def aggregate_e_a(args, e_a_mat_lst, grad_mat_lst, b_mat_lst, phi_mat_lst):
    """
        This function estimate the expectation of the e_a ((1/P)E(Psi^t*B*Phi)) using Monte Carlo average.
        :param args: Training settings.
        :param e_a_mat_lst: A list of matrices in size of N*N.
        :param grad_mat_lst: A list of matrices in size of P*N.
        :param b_mat_lst: A list of matrices in size of P*P.
        :param phi_mat_lst: A list of normal random matrices in size of P*N.
        :return:
    """
    for i in range(len(grad_mat_lst)):
        e_a_mat_lst[i].add_((1/(args.train_mc_iters*b_mat_lst[i].size()[0]))*torch.mm(torch.mm(
            torch.transpose(grad_mat_lst[i], 0, 1), b_mat_lst[i]), phi_mat_lst[i]))


def aggregate_e_b(args, e_b_mat_lst, grad_mat_lst, a_mat_lst, phi_mat_lst):
    """
        This function estimate the expectation of the e_b ((1/N)E(Phi^t*A*Psi)) using Monte Carlo average.
        :param args: Training settings.
        :param e_b_mat_lst: A list of matrices in size of P*P.
        :param grad_mat_lst: A list of matrices in size of P*N.
        :param a_mat_lst: A list of matrices in size of N*N.
        :param phi_mat_lst: A list of normal random matrices in size of P*N
        :return:
    """
    for i in range(len(grad_mat_lst)):
        e_b_mat_lst[i].add_((1/(args.train_mc_iters*a_mat_lst[i].size()[0]))*torch.mm(torch.mm(
            grad_mat_lst[i], a_mat_lst[i]), torch.transpose(phi_mat_lst[i], 0, 1)))


def update_m(m_mat_lst, a_mat_lst, b_mat_lst, avg_psi_mat_lst, eta=1, diagonal=False):
    """
        This function updates the mean according to M = M - B*B^t*E[Psi]*A*A^t.
        :param m_mat_lst: m_mat_lst: A list of matrices in size of P*N.
        :param a_mat_lst: A list of matrices in size of N*N.
        :param b_mat_lst: A list of matrices in size of P*P.
        :param avg_psi_mat_lst: A list of matrices in size of P*N.
        :param eta: .
        :param diagonal: .
        :return:
    """
    if diagonal:
        for i in range(len(m_mat_lst)):
            # M = M - diag(B*B^t)*E[Psi]*diag(A*A^t)
            m_mat_lst[i].copy_(torch.add(m_mat_lst[i], -eta,
                                         torch.mm(torch.mm(torch.diagflat(torch.diagonal(torch.mm(b_mat_lst[i], torch.transpose(b_mat_lst[i], 0, 1)))),
                                                           avg_psi_mat_lst[i]), torch.diagflat(torch.diagonal(torch.mm(a_mat_lst[i],
                                                                                         torch.transpose(a_mat_lst[i], 0, 1)
                                                                                     ))))))
    else:
        for i in range(len(m_mat_lst)):
            # M = M - B*B^t*E[Psi]*A*A^t
            m_mat_lst[i].copy_(torch.add(m_mat_lst[i], -eta,
                                         torch.mm(torch.mm(torch.mm(b_mat_lst[i], torch.transpose(b_mat_lst[i], 0, 1)),
                                                           avg_psi_mat_lst[i]), torch.mm(a_mat_lst[i],
                                                                                         torch.transpose(a_mat_lst[i], 0, 1)
                                                                                     ))))


def update_a_b(a_mat_lst, b_mat_lst, e_a_mat_lst, e_b_mat_lst, device, use_gsvd = False):
    """
        This function updates the matrices A & B using a solution to the non-linear matrix equation
        XX^{\top}+VEX^{\top}-V = 0.
        :param a_mat_lst:
        :param b_mat_lst:
        :param e_a_mat_lst:
        :param e_b_mat_lst:
        :return:
    """
    for i in range(len(a_mat_lst)):
        a_temp = solve_matrix_equation(torch.mm(a_mat_lst[i], torch.transpose(a_mat_lst[i], 0, 1)), e_a_mat_lst[i])
        b_temp = solve_matrix_equation(torch.mm(b_mat_lst[i], torch.transpose(b_mat_lst[i], 0, 1)), e_b_mat_lst[i])
        a_mat_lst[i].copy_(a_temp)
        b_mat_lst[i].copy_(b_temp)


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True