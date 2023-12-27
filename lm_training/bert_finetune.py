import sys

sys.path.append("/home/vmeshchaninov/DiffusionTextGeneration-cond-ca/")

import torch
import ml_collections
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning import seed_everything, Trainer

from lm_training.dataset_lightning import WikiDataModule, SSTDataModule
from lm_training.bert_lightning import BERTModel
from lm_training.util import Writer

import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 100.
    optim.linear_warmup = 1250
    optim.lr = 2e-5
    optim.min_lr = 2e-5
    optim.warmup_lr = 1e-8
    optim.weight_decay = 0.1
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6
    optim.precision = "32"

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 50_000
    training.training_iters = training.training_iters
    training.checkpoint_freq = 1_000_000
    training.eval_freq = 100
    training.batch_size = 32 // torch.cuda.device_count()

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 128
    data.num_workers = 16
    data.bert_recon_dataset = True

    model = config.model = ml_collections.ConfigDict()
    model.mlm_probability = 0.15
    model.pad_to_multiple_of = 3

    bert_config = config.bert_config = ml_collections.ConfigDict()
    bert_config.hidden_size = 768

    config.project_name = "lm-finetuning"
    config.checkpoint_name = "bert-training-768-0.15-None-2048-wiki_no_group"
    config.checkpoint_step = "130000"
    config.seed = 0
    config.hg_pretrain = False
    config.finetune = True
    config.model_name = "bert-base-uncased"
    config.exp_name = f"bert-finetune-{config.checkpoint_name}-{config.checkpoint_step}-{config.seed}"

    return config


def main():
    config = create_config()
    seed_everything(config.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    datamodule = SSTDataModule(
        tokenizer=tokenizer,
        config=config,
        collate_fn=None,
    )

    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
        )
    else:
        strategy = 'auto'

    trainer = Trainer(
        max_steps=config.training.training_iters,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.optim.grad_clip_norm,
        precision=config.optim.precision,
        strategy=strategy,
        enable_checkpointing=True,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./checkpoints/{config.exp_name}/",
                filename='step_{step:d}',
                every_n_train_steps=config.training.checkpoint_freq,
                save_top_k=-1,
                auto_insert_metric_name=False,
                save_weights_only=False
            ),
            LearningRateMonitor(logging_interval='step'),
            Writer(output_dir="pred_path", write_interval="epoch"),

        ],
        logger=WandbLogger(
            project=config.project_name,
            name=config.exp_name,
            config=config,
        ),
        val_check_interval=config.training.eval_freq,
        check_val_every_n_epoch=None
    )
    if config.hg_pretrain:
        model = BERTModel(config)
    else:
        model = BERTModel.load_from_checkpoint(
            checkpoint_path=f"./checkpoints/{config.checkpoint_name}/step_{config.checkpoint_step}.ckpt",
            config=config
        )
    model.model.cls = torch.nn.Linear(config.bert_config.hidden_size, 1)
    trainer.fit(model, datamodule=datamodule)

    # model = BERTModel.load_from_checkpoint(
    #     checkpoint_path="./checkpoints/bert-finetune-768-0.15-3-hg/step_2000.ckpt",
    #     config=config
    # )
    #
    # trainer.predict(model, datamodule=datamodule)


main()
