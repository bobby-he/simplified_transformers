"""Script for a training run."""

import hydra

import os
import logging

from datasets import load_dataset, DatasetDict
import transformers
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from simplified_transformers import model_utils, train_utils

log = logging.getLogger(__name__)


@hydra.main(config_path="simplified_transformers/config", config_name="config")
def launch(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    transformers.set_seed(cfg.seed)
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset(
        "huggingface-course/codeparrot-ds-valid", split="validation"
    )

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle(seed=0).select(
                range(cfg.num_token_mult * 100000)
            ),
            "valid": ds_valid.shuffle(seed=0).select(range(2000)),
        }
    )

    context_length = 128
    tokenizer = AutoTokenizer.from_pretrained(
        "huggingface-course/code-search-net-tokenizer", use_fast=True
    )

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    print(f"Input IDs length: {len(outputs['input_ids'])}")
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    model_config = AutoConfig.from_pretrained(
        cfg.model.name,
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        resid_pdrop=cfg.model.resid_pdrop,
        attn_pdrop=cfg.model.attn_pdrop,
        embd_pdrop=cfg.model.embd_pdrop,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        n_inner=int(cfg.model.n_embd * cfg.model.mlp_width_mult),
        initializer_range=cfg.model.initializer_range,
        output_attentions=cfg.report_attn_entropy,
        output_hidden_states=cfg.report_outliers,
    )
    model = GPT2LMHeadModel(model_config)

    model_config.update(
        {
            "attn_block_resid_gain": cfg.model.attn_block_resid_gain,
            "attn_block_skip_gain": cfg.model.attn_block_skip_gain,
            "mlp_block_resid_gain": cfg.model.mlp_block_resid_gain,
            "mlp_block_skip_gain": cfg.model.mlp_block_skip_gain,
            "attn_mat_resid_gain": cfg.model.attn_mat_resid_gain,
            "attn_mat_skip_gain": cfg.model.attn_mat_skip_gain,
            "value_resid_gain": cfg.model.value_resid_gain,
            "first_layer_value_resid_gain": cfg.model.first_layer_value_resid_gain,
            "value_skip_gain": cfg.model.value_skip_gain,
            "proj_resid_gain": cfg.model.proj_resid_gain,
            "last_layer_proj_resid_gain": cfg.model.last_layer_proj_resid_gain,
            "proj_skip_gain": cfg.model.proj_skip_gain,
            "trainable_attn_block_gains": cfg.model.trainable_attn_block_gains,
            "trainable_mlp_block_gains": cfg.model.trainable_mlp_block_gains,
            "trainable_attn_mat_gains": cfg.model.trainable_attn_mat_gains,
            "trainable_value_gains": cfg.model.trainable_value_gains,
            "trainable_proj_gains": cfg.model.trainable_proj_gains,
            "norm_type": cfg.model.norm_type,
            "val_proj_init_std": cfg.model.val_proj_init_std,
            "query_init_std": cfg.model.query_init_std,
            "key_init_std": cfg.model.key_init_std,
            "centre_attn": cfg.model.centre_attn,
            "centre_attn_gain": cfg.model.centre_attn_gain,
            "val_init_type": cfg.model.val_init_type,
            "proj_init_type": cfg.model.proj_init_type,
            "activation_function": cfg.model.activation_function,
            "lrelu_neg_slope": cfg.model.lrelu_neg_slope,
            "mlp_proj_init_std": cfg.model.mlp_proj_init_std,
            "parallel_layers": cfg.model.parallel_layers,
            "norm_position": cfg.model.norm_position,
            "tie_valproj_init": cfg.model.tie_valproj_init,
            "qk_norm_type": cfg.model.qk_norm_type,
            "dot_norm_type": cfg.model.dot_norm_type,
        }
    )

    model = model_utils.convertGPT2model(model, model_config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir="codeparrot-ds",
        per_device_train_batch_size=cfg.train.device_train_batch_size,
        per_device_eval_batch_size=cfg.train.device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        learning_rate=cfg.train.learning_rate,
        save_steps=10000,
        fp16=True,
        report_to="wandb" if cfg.use_wandb else "none",
        adam_epsilon=cfg.train.adam_epsilon,
        max_grad_norm=cfg.train.max_grad_norm,
    )

    # TODO: make this cleaner
    args.report_gains = cfg.report_gains
    args.report_attn_entropy = cfg.report_attn_entropy
    args.report_outliers = cfg.report_outliers

    trainer = train_utils.MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    trainer.train()


if __name__ == "__main__":
    launch()
