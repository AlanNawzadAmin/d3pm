import numpy as np
import torch
import torch.nn
from PIL import Image
from torchvision.utils import make_grid


def _at(a, t, x):
    # t is 1-d, x is integer value of 0 to num_classes - 1
    bs = t.shape[0]
    t = t.reshape((bs, *[1] * (x.dim() - 1)))
    return a[t, x, :]

def kls(dist1, dist2, eps): # KL of dists on last dim
    out = torch.softmax(dist1 + eps, dim=-1) * (
        torch.log_softmax(dist1 + eps, dim=-1)
        - torch.log_softmax(dist2 + eps, dim=-1)
    )
    return out.sum(dim=-1)#.mean()

def convert_to_distribution(x_0, num_classes, eps):
    # returns log probs of x_0 as a distribution
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_logits = torch.log(
            torch.nn.functional.one_hot(x_0, num_classes) + eps
        )
    else:
        x_0_logits = x_0.clone()
    return x_0_logits

def get_inf_gens(forward_kwargs, num_classes):
    if forward_kwargs['type'] == "uniform":
        L = torch.ones(num_classes, num_classes) / (num_classes-1)
        L.diagonal().fill_(-1)
    elif forward_kwargs['type'] == "gaussian":
        bandwidth = forward_kwargs['bandwidth']
        range_ = torch.arange(num_classes)
        diff_mat = (range_[:, None] - range_[None, :]) ** 2
        L = torch.exp(- diff_mat / (2 * (bandwidth * num_classes) ** 2))
        L = L / (L.sum(-1).max() - 1)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    if "normalize" in forward_kwargs.keys():
        if forward_kwargs['normalize']:
            L = L / (- L.diagonal()[:, None])
    return L

def get_gif(x, model, gen_trans_step, N, batch_size, gif_fname, img_fname):
    # save images
    cond = torch.arange(0, batch_size).to(x.device) % 10
    p = model.get_stationary()
    samples = torch.multinomial(p, num_samples=batch_size*x.shape[1:].numel(), replacement=True)
    init_noise = samples.reshape((batch_size,)+x.shape[1:]).to(x.device)

    images = model.sample_with_image_sequence(
        init_noise, cond, stride=3, trans_step=gen_trans_step,
    )
    # image sequences to gif
    gif = []
    for image in images:
        x_as_image = make_grid(image.float() / (N - 1), nrow=2)
        img = x_as_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gif.append(Image.fromarray(img))

    if gif_fname is not None:
        gif[0].save(
            gif_fname,
            save_all=True,
            append_images=gif[1:],
            duration=100,
            loop=0,
        )
    if img_fname is not None:
        last_img = gif[-1]
        last_img.save(img_fname)
