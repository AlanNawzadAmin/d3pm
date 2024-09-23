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
from pytorch_lightning.strategies import DDPStrategy

from d3pm_sc.ct_sched_cond import ScheduleCondition
from d3pm_sc.ct_sched_cond_sparse_k import ScheduleConditionSparseK
from d3pm_sc.masking_diffusion import MaskingDiffusion
from d3pm_sc.sedd import SEDD
from d3pm_sc.d3pm_classic import D3PMClassic
from d3pm_sc.discrete_sc import DiscreteScheduleCondition

from nets import get_model_setup
from data import get_dataloaders
from ema import EMA

import getpass

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

@hydra.main(version_base=None, config_path="configs", config_name="basic_language")
def train(cfg: DictConfig) -> None:
    wandb_key = "9e61d229e6b9dbfef3e2199c7e093a75bfe53135" if 'nvg' \
        in getpass.getuser() else "6a47f093d2a55e4f4e85b33767423f2db66355b8"
    wandb.login(key=wandb_key)
    ##### Load data
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    tokenizer = train_dataloader.tokenizer if hasattr(train_dataloader, "tokenizer") else None

    ##### Setup x0_model
    x0_model_class, nn_params = get_model_setup(cfg, tokenizer) 
    
    print(cfg)
    
    ##### Pick model
    model_name_dict = {"ScheduleCondition":ScheduleCondition,
                       "ScheduleConditionSparseK":ScheduleConditionSparseK,
                       "MaskingDiffusion":MaskingDiffusion,
                       "SEDD": SEDD,
                       "DiscreteScheduleCondition":DiscreteScheduleCondition,
                       "D3PMClassic":D3PMClassic}
    model = model_name_dict[cfg.model.model](
        x0_model_class,
        nn_params,
        num_classes=len(tokenizer) if tokenizer else cfg.data.N,
        hybrid_loss_coeff=cfg.model.hybrid_loss_coeff,
        gamma=cfg.model.gamma,
        forward_kwargs=OmegaConf.to_container(cfg.model.forward_kwargs, resolve=True),
        schedule_type=cfg.model.schedule_type,
        logistic_pars=cfg.model.logistic_pars,
        fix_x_t_bias=cfg.model.fix_x_t_bias,
        n_T=cfg.model.n_T,
        t_max=cfg.model.t_max,
        seed=cfg.model.seed,
        sedd_param=cfg.model.sedd_param,
        eff_num_classes=cfg.model.eff_num_classes,
        input_logits=cfg.model.input_logits,
        **OmegaConf.to_container(cfg.train, resolve=True),
    )

    ##### Load data
    model.pre_configure_model(train_dataloader)

    ##### Train
    wandb.init()
    wandb_logger = WandbLogger(project="debugging")
    lightning_model = model
    torch.set_float32_matmul_precision('high')

    # ddp = not cfg.model.model == "ScheduleConditionSparseK"
    trainer = Trainer(
        max_epochs=cfg.train.n_epoch, 
        accelerator='auto', 
        devices=torch.cuda.device_count(), 
        logger=wandb_logger, 
        # strategy="ddp",# if ddp else 'auto'
        strategy=DDPStrategy(broadcast_buffers=False),
        callbacks=[EMA(0.9999)]
    )
    trainer.fit(lightning_model, train_dataloader, test_dataloader)
    wandb.finish()

if __name__ == "__main__":
    train()
