import os
import json
import torch
import argparse
import torch.distributed as dist
import ml_collections
from datasets import disable_progress_bar
from transformers import BertConfig
import sys

from diffusion_holder import DiffusionRunner
from utils.util import set_seed, _BERT_SMALL
from estimation_utils.util import estimate_model, reduce_metrics, gather_texts
import diffusion_utils.schedulers as schedulers

from create_config import create_config
from estimation_utils.metrics import BloomMetric, RobertaMetric
from evaluate import load


def parse_option(config):
    parser = argparse.ArgumentParser("MMTD")
    if config.ddp:
        parser.add_argument('--local_rank', type=int, required=True)
    args, unparsed = parser.parse_known_args()
    return args

if __name__ == '__main__':
    model_name = 'qqp--t5-bert-cond_cfg_1751154_300000_'
    
    num_texts_ = 512 * 8
    batch_size_ = 1024

    metrics_json = dict()
    metrics_path = f"./metrics"
    os.makedirs(metrics_path, exist_ok=True)
    texts_path = "./generated_texts"
    os.makedirs(texts_path, exist_ok=True)

    config = create_config()
    if "base" in config.model.dif_enc_type:
        config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    else:
        config.bert_config = BertConfig(**_BERT_SMALL)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    
    config.local_rank = rank

    metric_bloom_fn = BloomMetric(device=f"cuda:{dist.get_rank()}")
    metric_roberta_fn = RobertaMetric(device=f"cuda:{dist.get_rank()}")
    metric_rouge_fn = load('/home/amshabalin/.cache/huggingface/metrics/rouge')

    config.checkpoints_prefix = model_name
    
    config.bert_config = BertConfig.from_pretrained("bert-base-uncased")
    config.bert_config.use_self_cond = config.use_self_cond
    config.bert_config.is_decoder = True
    config.bert_config.num_hidden_layers = 6

    seed = config.seed + dist.get_rank()
    set_seed(seed)

    num_texts_ = int(num_texts_ / dist.get_world_size())
    diffusion_ = DiffusionRunner(config, latent_mode=config.model.embeddings_type, eval=True)
    seed = config.seed + dist.get_rank()
    set_seed(seed)
    metrics, metrics_list, joint_texts, cond_texts, gen_texts, gt_texts = estimate_model(
        diffusion_, num_texts_, batch_size_, metric_bloom_fn, metric_roberta_fn, metric_rouge_fn
    )
    metrics_json[model_name] = reduce_metrics(metrics)
    cond_texts = gather_texts(cond_texts)
    gen_texts = gather_texts(gen_texts)
    gt_texts = gather_texts(gt_texts)

    if dist.get_rank() == 0:
        print(model_name)
        print(metrics)
        # print(f"Bloom metric: {metrics['Bloom metric']:0.5f}")
        # print(f"GPT2 metric: {metrics['GPT2 metric']:0.5f}")
        print(len(gt_texts))
        prefix = f"seq-len={config.data.max_sequence_len}-ode-{config.training.ode_sampling}-cosine"
        metrics_file = os.path.join(metrics_path, f"{model_name}-{prefix}.json")
        with open(metrics_file, "w") as file:
            json.dump(metrics_json, file)
        file_name = f"{texts_path}/{model_name}-{prefix}.txt"
        with open(file_name, "w") as file:
            for i in range(len(gt_texts)):
                print('COND:', cond_texts[i], file=file)
                print('GEN:', gen_texts[i], file=file)
                print('GT:', gt_texts[i], file=file)
                print('', file=file)
        print(file_name)
