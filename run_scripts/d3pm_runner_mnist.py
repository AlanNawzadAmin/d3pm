import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb
from d3pm_sc.d3pm import D3PM
from d3pm_sc.unet import UNet

if __name__ == "__main__":

    n_T = 1000
    N = 2  # number of classes for discretized state per pixel
    
    uniform_noise = True
    schedule_conditioning = False

    d3pm = D3PM(
        UNet(1, N, n_T), 
        n_T, 
        num_classes=N, 
        hybrid_loss_coeff=0.0, 
        schedule_conditioning=schedule_conditioning
    ).cuda()

    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    dataset = MNIST(
        "/home/nvg7279/d3pm/data",
        train=True,
        # download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=32)

    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=1e-3)
    d3pm.train()

    n_epoch = 400
    device = "cuda"

    global_step = 0
    for i in range(n_epoch):

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, cond in pbar:
            optim.zero_grad()
            x = x.to(device)
            cond = cond.to(device)

            # discritize x to N bins
            x = (x * (N - 1)).round().long().clamp(0, N - 1)
            loss, info = d3pm(x, cond)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 0.1)

            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
            pbar.set_description(
                f"loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}"
            )
            optim.step()
            global_step += 1

            if global_step % 300 == 1:
                d3pm.eval()

                with torch.no_grad():
                    cond = torch.arange(0, 4).cuda() % 10
                    if uniform_noise:
                        init_noise = torch.randint(0, N, (4, 1, 32, 32)).cuda()
                    else:
                        pass

                    if schedule_conditioning:
                        S = sample_n_transitions(
                            d3pm.beta_t, 
                            init_noise[0].flatten().shape[0], 
                            torch.tensor(4 * [n_T - 1])
                        )
                        S = S.swapaxes(0, 1).reshape(*init_noise.shape).long()
                        S = S.cuda()
                    else:
                        S = None

                    images = d3pm.sample_with_image_sequence(
                        init_noise, cond, S, stride=40
                    )
                    # image sequences to gif
                    gif = []
                    for image in images:
                        x_as_image = make_grid(image.float() / (N - 1), nrow=2)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))

                    gif[0].save(
                        f"contents/sample_{global_step}.gif",
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )

                    last_img = gif[-1]
                    last_img.save(f"contents/sample_{global_step}_last.png")

                d3pm.train()