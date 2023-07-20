import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _non_saturating_loss(D_real_logits, D_gen_logits, D_real=None, D_gen=None):
    D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,
                                                     target=torch.ones_like(D_real_logits))
    D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,
                                                    target=torch.zeros_like(D_gen_logits))
    D_loss = D_loss_real + D_loss_gen

    G_loss = F.binary_cross_entropy_with_logits(input=D_gen_logits,
                                                target=torch.ones_like(D_gen_logits))

    return D_loss, G_loss


def _least_squares_loss(D_real, D_gen, D_real_logits=None, D_gen_logits=None):
    D_loss_real = torch.mean(torch.square(D_real - 1.0))
    D_loss_gen = torch.mean(torch.square(D_gen))
    D_loss = 0.5 * (D_loss_real + D_loss_gen)

    G_loss = 0.5 * torch.mean(torch.square(D_gen - 1.0))

    return D_loss, G_loss

def relativistic_least_squares_loss(D_real, D_gen, D_real_logits=None, D_gen_logits=None):
    Relativisitc_loss_real = D_real - D_gen.mean(0, keepdim=True)
    Relativisitc_loss_gen = D_gen - D_real.mean(0, keepdim=True)

    D_loss = torch.mean(torch.square(Relativisitc_loss_real - 1.0)) + torch.mean(torch.square(Relativisitc_loss_gen + 1.0))

    G_loss = torch.mean(torch.square(Relativisitc_loss_real + 1.0)) + torch.mean(torch.square(Relativisitc_loss_gen - 1.0))

    return D_loss, G_loss

def gan_loss(gan_loss_type, disc_out, mode='generator_loss'):
    if gan_loss_type == 'non_saturating':
        loss_fn = _non_saturating_loss
    elif gan_loss_type == 'least_squares':
        loss_fn = _least_squares_loss
    elif gan_loss_type == 'relative_least_squares':
        loss_fn = relativistic_least_squares_loss
    else:
        raise ValueError('Invalid GAN loss')

    D_loss, G_loss = loss_fn(D_real=disc_out.D_real, D_gen=disc_out.D_gen,
                             D_real_logits=disc_out.D_real_logits, D_gen_logits=disc_out.D_gen_logits)

    loss = G_loss if mode == 'generator_loss' else D_loss

    return loss
