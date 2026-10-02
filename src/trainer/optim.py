"""
Optimizer and schedulers
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_optimizer(optim: str, model: nn.Module, **kwargs: any) -> Optimizer:
    """
    Return training optimizer
    """
    if 'betas' in kwargs:  # A list in the YAML config; torch expects a tuple
        kwargs['betas'] = tuple(kwargs['betas'])
    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optim == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif optim in ['adamw', 'adamw_torch']:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    elif optim == 'adamw_torch_fused':
        return torch.optim.AdamW(model.parameters(), **kwargs, fused=True)
    elif optim == 'adafactor':
        from transformers import Adafactor
        kwargs['relative_step'] = False  # for now
        return Adafactor(model.parameters(), **kwargs)
    else:
        raise NotImplementedError(f"{optim} optimizer not implemented sorry.")


def get_scheduler(lr_scheduler_type: str, optimizer: Optimizer, 
                  **kwargs: any) -> LRScheduler:
    """
    Return learning rate scheduler
    """
    if lr_scheduler_type in ['plateau', 'reduce_lr_on_plateau']:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer=optimizer, **kwargs)
    
    elif lr_scheduler_type == 'cosine_warmup':
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer=optimizer, **kwargs)

    elif lr_scheduler_type == 'cosine_warmup_min_lr':  # Cosine decay to min_lr_rate x the peak, not to 0
        from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
        return get_cosine_with_min_lr_schedule_with_warmup(optimizer=optimizer, **kwargs)
    
    elif lr_scheduler_type in ['linear_warmup', 'linear']:
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(optimizer=optimizer, **kwargs)
    
    else:
        return None