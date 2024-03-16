# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LACE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys
sys.path.append('boostPM_py')
import torch
import torch.nn as nn
import boostPM_py
import rpy2.robjects as robj
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_normal


# from torchdiffeq import odeint_adjoint
# from torchdiffeq import odeint as odeint_normal

# import dnnlib, legacy
# from latent_model import DenseEmbedder
# from models import DenseNet
# from models import WideResNet
# from utils import torch2rmat

# NETWORK_PKL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/cifar10u-cifar-ada-best-fid.pkl'
normal_dist = torch.distributions.normal.Normal(0., 1.)
readRDS = robj.r['readRDS']
g_path = '/hpc/home/zw122/tree_condsamp/LACE/tree-toy/pretrained/gaussianout.rds'
out = readRDS(g_path)


class F(nn.Module):
    def __init__(self, x_space, latent_dim, n_classes):
        # self.f: classifier, g(z) -> logits
        # self.g: pre-trained generator, a mapping z->g(z) or z->w or g->g

        super(F, self).__init__()
        self.x_space = x_space

        if x_space == 'toy_i':
            def mapping_z_to_i(z):
                gz = boostPM_py.generator(out, z)
                gz.requires_grad = True
                return gz
            self.g = mapping_z_to_i

        else:
            raise NotImplementedError('unknown x_space, choices: [cifar10_i, toy_z]')

    def classify(self, z):
        logits = self.f(self.g(z)).squeeze()
        # logits = self.f(self.g(z))
        return logits

    def classify_x(self, x):
        logits = self.f(x).squeeze()
        # logits = self.f(x)
        return logits

    def generate_images(self, g_z_sampled, is_detached=True):
        """Generate images from g_z_sampled or dataset"""

        # Synthesize the result from i
        if self.x_space == 'toy_i':
            img = g_z_sampled
            return img.detach() if is_detached else img
        return None


class CCF(F):
    def __init__(self, x_space, latent_dim, n_classes):
        super(CCF, self).__init__(x_space, latent_dim=latent_dim, n_classes = n_classes)

        self.x_space = x_space
        self.d = {}
        print(f'Working in the x_space: {x_space}')
    def classify_with_gz(self, z):
        gz = self.g(z)
        logits = self.f(gz).squeeze()
        # logits = self.f(gz)
        return logits, gz
    def get_cond_energy(self, z, y):
        logits, gz = self.classify_with_gz(z)
        self.d['gz'] = gz
        energy_output = torch.gather(logits, 1, y[:, None]).squeeze() - logits.logsumexp(1)
        return energy_output

    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y) - torch.linalg.norm(z, dim=1) ** 2 * 0.5
        return energy_output


# ----------------------------------------------------------------------------

_sample_q_dict = dict()  # name => fn


def register_sample_q(fn):
    assert callable(fn)
    _sample_q_dict[fn.__name__] = fn
    return fn


@register_sample_q
def sample_q_sgld(ccf, y, device, save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    sample_path = []
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    sgld_lr = kwargs['sgld_lr']
    sgld_std = kwargs['sgld_std']
    n_steps = kwargs['n_steps']

    # generate initial samples
    init_sample = torch.randn(y.size(0), latent_dim).to(device)
    z_k = torch.autograd.Variable(init_sample, requires_grad=True)

    # sgld
    for k in range(n_steps):
        if k%(n_steps//10) == 0:
            print(f'Running SGLD step {k}', flush = True)
        energy_neg = ccf(z=z_k, y=y)
        gz = ccf.d['gz']
        dE_dg = torch.autograd.grad(energy_neg.sum(), [gz])[0]
        z_k.requires_grad = False
        diag_log_jac = boostPM_py.diag_log_dg_dz(out, z_k)
        assert dE_dg.shape == diag_log_jac.shape
        diag_dg_dz = torch.exp(diag_log_jac)
        f_prime = dE_dg * diag_dg_dz - z_k# element wise
        z_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(z_k)
        if save_path is not None and k % every_n_plot == 0:
            x_sampled = ccf.generate_images(gz) # identity
            # plot('{}/samples_class{}_nsteps{}.png'.format(save_path, y[0].item(), k), x_sampled)
            sample_path.append(z_k.clone())
    ccf.train()
    final_samples = z_k.detach()
    return sample_path, final_samples


class VPODE(nn.Module):
    def __init__(self, ccf, y, beta_min=0.1, beta_max=20, T=1.0, save_path=None, plot=None, every_n_plot=5):
        super().__init__()
        self.ccf = ccf
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.T = T
        self.y = y
        self.save_path = save_path
        self.n_evals = 0
        self.every_n_plot = every_n_plot
        self.plot = plot

    def forward(self, t_k, states):
        z = states[0]

        if self.save_path is not None and self.n_evals % self.every_n_plot == 0:
            energy_neg = self.ccf(z=z.detach(), y=self.y)
            g_z_sampled = self.ccf.d['gz']
            # g_z_sampled = self.ccf.g(z.detach())
            x_sampled = self.ccf.generate_images(g_z_sampled)
            # self.plot(f'{self.save_path}/samples_cls{self.y[0].item()}_nsteps{self.n_evals:03d}_tk{t_k}.png',
                    #   x_sampled)

        # with torch.set_grad_enabled(True):
            # z.requires_grad_(False)
        beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
        # cond_energy_neg = self.ccf.get_cond_energy(z, self.y)
        cond_energy_neg = self.ccf.get_cond_energy(z, self.y)
        # cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]
        gz = self.ccf.d['gz']
        dE_dg = torch.autograd.grad(cond_energy_neg.sum(), [gz])[0]
        # z.requires_grad = False
        diag_log_jac = boostPM_py.diag_log_dg_dz(out, z.detach())
        assert dE_dg.shape == diag_log_jac.shape
        diag_dg_dz = torch.exp(diag_log_jac)
        cond_f_prime = dE_dg * diag_dg_dz 
        dz_dt = -0.5 * beta_t * cond_f_prime

        self.n_evals += 1

        return dz_dt,


@register_sample_q
def sample_q_ode(ccf, y, device, save_path=None, plot=None, every_n_plot=5, **kwargs):
    """sampling in the z space"""
    ccf.eval()

    latent_dim = kwargs['latent_dim']
    atol = kwargs['atol']
    rtol = kwargs['rtol']
    method = kwargs['method']
    use_adjoint = kwargs['use_adjoint']
    beta_0 = kwargs.get('beta_0', 0.1)
    beta_1 = kwargs.get('beta_1', 20)
    T = kwargs.get('T', 1.0)

    # generate initial samples
    z_k = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)

    # ODE function
    vpode = VPODE(ccf, y, save_path=save_path, plot=plot, every_n_plot=every_n_plot, beta_min = beta_0, beta_max = beta_1, T = T)
    states = (z_k,)
    integration_times = torch.linspace(vpode.T, 0., 2).type(torch.float32).to(device)

    # ODE solver
    odeint = odeint_adjoint if use_adjoint else odeint_normal
    state_t = odeint(
        vpode,
        states,
        integration_times,
        atol=atol,
        rtol=rtol,
        method=method)

    ccf.train()
    z_t0 = state_t[0][-1]
    print(f'#ODE steps for {y[0].item()}: {vpode.n_evals}')

    return z_t0.detach()


# @register_sample_q
# def sample_q_vpsde(ccf, y, device=torch.device('cuda'), save_path=None, plot=None, every_n_plot=5,
#                    beta_min=0.1, beta_max=20, T=1, eps=1e-3, **kwargs):
#     """sampling in the z space"""
#     ccf.eval()

#     latent_dim = kwargs['latent_dim']
#     N = kwargs['N']
#     correct_nsteps = kwargs['correct_nsteps']
#     target_snr = kwargs['target_snr']

#     # generate initial samples
#     z_init = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
#     z_k = torch.autograd.Variable(z_init, requires_grad=True)

#     discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
#     alphas = 1. - discrete_betas
#     timesteps = torch.linspace(T, eps, N, device=device)

#     # vpsde
#     for k in range(N):

#         if save_path is not None and k % every_n_plot == 0:
#             g_z_sampled = ccf.g(z_k.detach())
#             x_sampled = ccf.generate_images(g_z_sampled)
#             plot('{}/samples_class{}_nsteps{}.png'.format(save_path, y[0].item(), k), x_sampled)

#         energy_neg = ccf(z_k, y=y)

#         # predictor
#         t_k = timesteps[k]
#         timestep = (t_k * (N - 1) / T).long()
#         beta_t = discrete_betas[timestep]
#         alpha_t = alphas[timestep]

#         score_t = torch.autograd.grad(energy_neg.sum(), [z_k])[0]

#         z_k = (2 - torch.sqrt(alpha_t)) * z_k + beta_t * score_t
#         noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)
#         z_k = z_k + torch.sqrt(beta_t) * noise

#         # corrector
#         for j in range(correct_nsteps):
#             noise = torch.FloatTensor(y.size(0), latent_dim).normal_(0, 1).to(device)

#             grad_norm = torch.norm(score_t.reshape(score_t.shape[0], -1), dim=-1).mean()
#             noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
#             step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha_t

#             assert step_size.ndim == 0, step_size.ndim

#             z_k_mean = z_k + step_size * score_t
#             z_k = z_k_mean + torch.sqrt(step_size * 2) * noise

#     ccf.train()
#     final_samples = z_k.detach()

#     return final_samples
