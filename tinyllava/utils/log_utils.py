import logging
import os
import sys

import torch.distributed as dist


root_logger = None

# def print_rank0(*args):
#     local_rank = dist.get_rank()
#     if local_rank == 0:
#         print(*args)

def print_rank0(*args, **kwargs):
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        rank = 0
    if rank == 0:
        print(*args, **kwargs, flush=True)


def logger_setting(save_dir=None):
    global root_logger
    if root_logger is not None:
        return root_logger
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, 'log.txt')
            if not os.path.exists(save_file):
                os.system(f"touch {save_file}")
            fh = logging.FileHandler(save_file, mode='a')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
            return root_logger
        
# def log(*args):
#     global root_logger
#     # local_rank = dist.get_rank()
#     local_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
#     if local_rank == 0:
#         root_logger.info(*args)

def log(*args, **kwargs):
    """
    Safe logger:
    - works when torch.distributed is not initialized (single process)
    - works when root_logger is None (not yet initialized)
    """
    # rank
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:
        rank = 0

    # only rank0 prints / logs
    if rank != 0:
        return

    msg = " ".join(str(a) for a in args)

    global root_logger
    if root_logger is not None:
        try:
            root_logger.info(msg)
            return
        except Exception:
            pass

    # fallback: print to stdout
    print(msg, flush=True)



        
def log_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Total Parameters: {total_params}, Total Trainable Parameters: {total_trainable_params}')
    log(f'Trainable Parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank0(f"{name}: {param.numel()} parameters")
