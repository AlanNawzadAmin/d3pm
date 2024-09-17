
from omegaconf import OmegaConf

from d3pm_sc.unet import UNet, KingmaUNet, SimpleUNet
from d3pm_sc.dit_vision import DiT_Llama
from d3pm_sc.dit_text import DIT

image_nn_name_dict = {
    "SimpleUNet":SimpleUNet,
    "KingmaUNet":KingmaUNet,
    "UNet":UNet,
    "DiT_Llama":DiT_Llama
}

text_nn_name_dict = {
    "DIT": DIT
}

def get_model_setup(cfg, tokenizer=None):
    if cfg.architecture.x0_model_class in image_nn_name_dict:
        schedule_conditioning = cfg.model.model in [
            "ScheduleCondition", "DiscreteScheduleCondition"
        ]
        nn_params = cfg.architecture.nn_params
        nn_params = (OmegaConf.to_container(nn_params, resolve=True)
                    if nn_params is not None else {})
        nn_params = {
            "n_channel": 1 if cfg.data.data == 'MNIST' else 3, 
            "N": cfg.data.N + (cfg.model.model == 'MaskingDiffusion'),
            "n_T": cfg.model.n_T,
            "schedule_conditioning": schedule_conditioning,
            "s_dim": cfg.architecture.s_dim,
            **nn_params
        }
    
        return image_nn_name_dict[cfg.architecture.x0_model_class], nn_params
    
    elif cfg.architecture.x0_model_class in text_nn_name_dict:
        schedule_conditioning = cfg.model.model in [
            "ScheduleCondition", "DiscreteScheduleCondition"
        ]
        nn_params = cfg.architecture.nn_params
        nn_params = (OmegaConf.to_container(nn_params, resolve=True)
                    if nn_params is not None else {})
        
        nn_params["n_T"] = cfg.model.n_T
        nn_params["s_dim"] = cfg.architecture.s_dim
        
        print(nn_params)
        
        nn_params = {
            "config": nn_params,
            "vocab_size": len(tokenizer),
            "schedule_conditioning": schedule_conditioning,
        }
        
        return text_nn_name_dict[cfg.architecture.x0_model_class], nn_params