import sys
import numpy as np
import torch
import torch.nn as nn
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from d3pm_sc.unet import UNet, SimpleUNet
from d3pm_sc.dit import DiT_Llama

from d3pm_sc.ct_sched_cond import ScheduleCondition
from d3pm_sc.masking_diffusion import MaskingDiffusion
from d3pm_sc.d3pm_classic import D3PMClassic
from d3pm_sc.discrete_sc import DiscreteScheduleCondition

@hydra.main(version_base=None, config_path="configs", config_name="basic")
def train(cfg: DictConfig) -> None:
    ##### Setup x0_model
    schedule_conditioning = cfg.model.model in ["ScheduleCondition", "DiscreteScheduleCondition"]
    nn_params = cfg.architecture.nn_params
    nn_params = (OmegaConf.to_container(nn_params, resolve=True)
                 if nn_params is not None else {})
    nn_params = {"n_channel": 1 if cfg.data.data == 'MNIST' else 3, 
                 "N": cfg.data.N,
                 "n_T": cfg.model.n_T,
                 "schedule_conditioning": schedule_conditioning,
                 "s_dim": cfg.architecture.s_dim,
                 **nn_params
                }
    nn_name_dict = {"SimpleUNet":SimpleUNet,
                    "UNet":UNet,
                    "DiT_Llama":DiT_Llama}
    x0_model_class = nn_name_dict[cfg.architecture.x0_model_class]

    
    ##### Pick model
    model_name_dict = {"ScheduleCondition":ScheduleCondition,
                       "MaskingDiffusion":MaskingDiffusion,
                       "DiscreteScheduleCondition":DiscreteScheduleCondition,
                       "D3PMClassic":D3PMClassic}
    model = model_name_dict[cfg.model.model](
        x0_model_class,
        nn_params,
        num_classes=cfg.data.N,
        hybrid_loss_coeff=cfg.model.hybrid_loss_coeff,
        gamma=cfg.model.gamma,
        forward_kwargs=OmegaConf.to_container(cfg.model.forward_kwargs, resolve=True),
        logistic_pars=cfg.model.logistic_pars,
        fix_x_t_bias=cfg.model.fix_x_t_bias,
        n_T=cfg.model.n_T,
        lr=cfg.train.lr,
        grad_clip_val=cfg.train.grad_clip_val,
        weight_decay=cfg.train.weight_decay,
    )

    ##### Load data
    batch_size = cfg.train.batch_size
    data_name_dict = {"CIFAR10":CIFAR10, "MNIST":MNIST}
    dataset = data_name_dict[cfg.data.data](
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    def collate_fn(batch):
        x, cond = zip(*batch)
        x = torch.stack(x)
        cond = torch.tensor(cond)
        cond = cond * cfg.data.conditional
        x = (x * (cfg.data.N - 1)).round().long().clamp(0, cfg.data.N - 1)
        return x, cond
    train_size = int(len(dataset) * 0.9)
    dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn)

    ##### Train
    wandb_logger = WandbLogger(project="debugging")
    lightning_model = model
    
    trainer = Trainer(max_epochs=cfg.train.n_epoch, accelerator='auto', devices='auto', logger=wandb_logger)
    trainer.fit(lightning_model, dataloader, test_dataloader)
    wandb.finish()

if __name__ == "__main__":
    train()
