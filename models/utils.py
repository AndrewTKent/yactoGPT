# models/utils.py
import torch
from typing import Dict, Any


def configure_optimizers(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple = (0.9, 0.95),
    device_type: str = 'cuda'
) -> torch.optim.Optimizer:
    """Configure AdamW optimizer with weight decay fix.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        device_type: Device type for fused kernel
        
    Returns:
        Configured optimizer
    """
    # Separate parameters that should and shouldn't have weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
                
    # Special case position embedding
    no_decay.add('pos_emb')
    
    # Validate that all parameters are accounted for
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"
    
    # Create optimizer groups
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    # Use fused AdamW if available
    use_fused = (device_type == 'cuda') and ('fused' in torch.optim.AdamW.__init__.__code__.co_varnames)
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        fused=use_fused
    )
    
    return optimizer