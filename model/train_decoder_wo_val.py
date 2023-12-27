import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, RobertaTokenizerFast, T5TokenizerFast, ElectraTokenizerFast

import os
import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

from data.dataset_clean_wiki import WikipediaCleanDatasetUnconditional

from utils.util import dict_to_cuda

from model.bert_encoder import BertEncoderModel
from model.t5_encoder import T5EncoderModel
from model.roberta_encoder import RobertaEncoderModel
from model.electra_encoder import ElectraEncoderModel
from model.emb_encoder import EmbEncoderModel
from model.decoder import Decoder


def reconstruction_loss(target, prediction_scores, mask):
    if mask is None:
        return cross_entropy(
            input=prediction_scores.view(-1, prediction_scores.shape[-1]),
            target=target.view(-1),
        )

    ce_losses = cross_entropy(
        input=prediction_scores.view(-1, prediction_scores.shape[-1]),
        target=target.view(-1),
        reduce=False,
    )
    ce_losses = ce_losses * mask.reshape(-1)
    ce_loss = torch.sum(ce_losses) / torch.sum(mask)
    return ce_loss


def train(encoder, decoder, tokenizer, tokenizer_gen, exp_name):
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")

    max_sequence_len = 128
    batch_size = 1024
    # train_dataset = create_wiki_dataset()

    train_dataset = next(WikipediaCleanDatasetUnconditional(
        split="train",
        tokenizer=tokenizer,
        max_sequence_len=max_sequence_len,
    ).get_data())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=2e-4,
        weight_decay=0.001,
        # eps=1e-6,
        betas=(0.9, 0.98),
    )

    eval_freq = 100
    eval_mode = False
    step = 0
    epochs = 1
    for epoch in range(epochs):
        decoder.train()
        T = tqdm(train_loader)
        for X in T:
            if (step % eval_freq) == 0:
                eval_mode = True
            if eval_mode:
                decoder.eval()
            else:
                step += 1

            text = tokenizer.batch_decode(X["input_ids"], skip_special_tokens=True)

            X = tokenizer_gen(
                text=text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=max_sequence_len,
                return_tensors="pt",
            )

            X = dict_to_cuda(X)
            targets = X["input_ids"].type(torch.LongTensor).cuda()
            mask = X["attention_mask"]
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    emb = encoder(**{
                        "input_ids": X["input_ids"],
                        "attention_mask": X["attention_mask"]
                    })

            if not eval_mode:
                sigma = 0.1
                eps = torch.randn_like(emb) * sigma
                emb = emb + eps

            logits = decoder(emb)

            loss = reconstruction_loss(targets, logits, mask=None)
            if not eval_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(),
                    max_norm=1.0
                )
                optimizer.step()

            tokens = logits.argmax(dim=-1)
            acc = torch.mean((targets == tokens) * 1.)
            if not eval_mode:
                wandb.log({f'train loss': loss.item()}, step=step)
                wandb.log({f'train accuracy': acc.item()}, step=step)
            else:
                wandb.log({f'valid loss': loss.item()}, step=step)
                wandb.log({f'valid accuracy': acc.item()}, step=step)

            T.set_description(f"Loss: {loss.item():0.6f}")
            if step > 5000:
                break

            if eval_mode:
                decoder.train()
                eval_mode = False
                step += 1

    checkpoints_folder = './checkpoints/'
    name = os.path.join(checkpoints_folder, f"decoder-{exp_name}.pth")
    decoder.eval()
    torch.save(
        {
            "decoder": decoder.module.state_dict(),
        },
        name
    )
    print(f"Save model to: {name}")


def main():
    bert_cfg = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(bert_cfg)

    cfg = "bert-base-uncased"
    tokenizer_gen = BertTokenizerFast.from_pretrained(cfg)

    from model.bert_encoder_llm import BertEncoderModel
    encoder = BertEncoderModel.from_pretrained(
        "./lm_training/checkpoints/bert-training-768-120-0.15-None-2048-wiki_no_group/bert-150000/",
        enc_normalizer=None
    ).eval().cuda()

    decoder = torch.nn.DataParallel(
        Decoder(
            input_size=120,
            hidden_size=encoder.config.hidden_size,
            vocab_size=encoder.config.vocab_size
        )
    ).train().cuda()

    encoder = torch.nn.DataParallel(encoder)

    exp_name = "my_bert-768-120-150000"
    wandb.init(project="decoders", name=exp_name, mode="online")
    train(encoder, decoder, tokenizer, tokenizer_gen, exp_name=exp_name)


main()
