import os
import sys
import torch
import psutil
import datasets
import argparse
import torch.distributed as dist
from datasets import disable_progress_bar
from datasets.utils.logging import set_verbosity_error
from transformers import BertConfig

datasets.config.IN_MEMORY_MAX_SIZE = psutil.virtual_memory().available

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL
from create_config import create_config

# disable_progress_bar()
# set_verbosity_error()

sys.path.append("/home/amshabalin/DiffusionTextGeneration-cond-ca/")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


if __name__ == '__main__':
    config = create_config()
    suffix = f"t5small-bert-cond_cfg_6l"
    config.checkpoints_prefix = f"{config.model.dataset}-" \
                                f"{config.model.downstream_task if config.model.downstream_task is not None else ''}-" \
                                f"train_cond_enc-" \
                                f"{suffix}_{os.environ.get('SLURM_JOB_ID')}"
    
    if "base" in config.model.dif_enc_type:
        config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    else:
        config.bert_config = BertConfig(**_BERT_SMALL)
       
    config.bert_config.use_self_cond = config.use_self_cond
    config.bert_config.is_decoder = True
    config.bert_config.num_hidden_layers = 6

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    config.local_rank = rank
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()
    seed = config.seed
    set_seed(seed)
    # os.environ['CONFIG_PATH'] = "/home/vmeshchaninov/DiffusionTextGeneration/data/config.json"
    if dist.get_rank() == 0:
        print(config)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type)

    seed = config.seed + dist.get_rank()
    set_seed(seed)
    diffusion.train(
        project_name=config.project_name,
        experiment_name=config.checkpoints_prefix
    )
