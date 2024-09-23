import io
import tempfile
import numpy as np
import pytorch_lightning as pl
import wandb
import torch
import torch.optim as optim
from PIL import Image
from torchvision.utils import make_grid
import torch.nn.functional as F
from tqdm import tqdm

def get_gif(sample_x, model, gen_trans_step, batch_size):
    # save images
    cond = torch.arange(0, batch_size).to(sample_x.device) % 10
    p = model.get_stationary()
    samples = torch.multinomial(p, num_samples=batch_size*sample_x.shape[1:].numel(), replacement=True)
    init_noise = samples.reshape((batch_size,)+sample_x.shape[1:]).to(sample_x.device)

    images = model.sample_with_image_sequence(
        init_noise, cond, stride=3, n_T=gen_trans_step,
    )
    if images is not None:
        # image sequences to gif
        gif = []
        for image in images:
            x_as_image = make_grid(image.float() / (model.num_classes - 1), nrow=2)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))
    
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
            gif[0].save(
                temp_file.name,
                format='GIF',
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file_img:
            last_img = gif[-1]
            last_img.save(temp_file_img)
        return temp_file.name, temp_file_img.name
    else: 
        return None, None
    

class DiffusionTrainer(pl.LightningModule):
    def __init__(self, lr=1e-3, gen_trans_step=200, n_gen_images=4, grad_clip_val=1, weight_decay=0, seed=0, n_stat_samples=2e6, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.weight_decay = weight_decay
        # logging
        self.sample_x = None
        self.validation_step_outputs = []
        self.gen_trans_step = gen_trans_step
        self.n_gen_images = n_gen_images
        self.n_stat_samples = n_stat_samples

    def forward(self, x):
        return NotImplementedError

    def get_kl_t1(self, x):
        return NotImplementedError

    def pre_configure_model(self, dataloader):
        pass

    def calc_p0(self, dataloader):
        # get stationary dist
        p0 = torch.ones(self.num_classes)
        pbar = tqdm(total=self.n_stat_samples)
        for i, batch in tqdm(enumerate(dataloader)):
            if p0.sum() > self.n_stat_samples:  
                break
            if isinstance(batch, tuple): #image datasets
                x, _ = batch
            elif isinstance(batch, dict): #text datasets
                x = batch['input_ids']
            new =  F.one_hot(x.long(), num_classes=self.num_classes).float().view((-1, self.num_classes)).sum(0)
            p0 = p0 + new
            pbar.update(new.sum().item())
        pbar.close()
        p0 = p0 / p0.sum()
        self.p0 = p0
        # self.register_buffer("p0", p0)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple): #image datasets
            x, cond = batch
            if cond is not None:
                if cond.dim() == x.dim(): #protein datasets
                    attn_mask = cond
                    cond = None
        elif isinstance(batch, dict): #text datasets
            x, attn_mask = batch['input_ids'], batch['attention_mask']
            cond = batch['cond'] if 'cond' in batch else None
        # x, cond = x.to(device), cond.to(device)
        loss, info = self(x, cond, attn_mask)
        if self.sample_x is None:
            self.sample_x = x[:1]
        self.log('train_loss', info['vb_loss'], sync_dist=True)
        self.log('train_ce_loss', info['ce_loss'], sync_dist=True)
        # with torch.no_grad():
        #     param_norm = sum([torch.norm(p) for p in self.parameters()])
        # self.log('param_norm', param_norm, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):        
        if isinstance(batch, tuple): #image datasets
            x, cond = batch
            attn_mask = None
            if cond is not None:
                if cond.dim() == x.dim(): #protein datasets
                    attn_mask = cond
                    cond = None
        elif isinstance(batch, dict): #text datasets
            x, attn_mask = batch['input_ids'], batch['attention_mask']
            cond = batch['cond'] if 'cond' in batch else None

        # x, cond = x.to(device), cond.to(device)
        loss, info = self(x, cond, attn_mask)
        self.log('val_l01', info['vb_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_l1', self.get_kl_t1(x).detach().item(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_ce_loss', info['ce_loss'], on_step=False, on_epoch=True, sync_dist=True)
        loss_dict = {
            "val_ce_loss": info['ce_loss'],
            "val_l01": info['vb_loss'],
            "val_l1": self.get_kl_t1(x).detach().item()
        }
        return loss_dict

    def on_validation_epoch_end(self,):
        # generate image
        if self.sample_x is not None:
            with torch.no_grad():
                gif_fname, img_fname = get_gif(self.sample_x, self, self.gen_trans_step, self.n_gen_images)
            if gif_fname is not None:
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    wandb.log({"sample_gif": wandb.Image(gif_fname)})
                    wandb.log({"sample_gif_last": wandb.Image(img_fname)})

    def on_fit_start(self):
        if isinstance(self.logger, pl.loggers.WandbLogger):
            wandb.config.update(self.hparams)

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_val)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": self.grad_clip_val,
            "weight_decay": self.weight_decay,
            "gradient_clip_algorithm": "norm"
        }
