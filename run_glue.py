# coding=utf-8
# This file has been modified by ASAPP. The original file is licensed under the 
# Apache License Version 2.0.  The modifications by ASAPP are licensed under
# the MIT license.
# 
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.optim import Adam
from options import get_parser
from model_utils import ElectraForSequenceClassification

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from prior_wd_optim import PriorWD

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer_grouped_parameters(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    if args.layerwise_learning_rate_decay == 1.0:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "lr": args.learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": args.learning_rate,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
                "weight_decay": 0.0,
                "lr": args.learning_rate,
            },
        ]

        if args.model_type in ["bert", "roberta", "electra"]:
            num_layers = model.config.num_hidden_layers
            layers = [getattr(model, args.model_type).embeddings] + list(getattr(model, args.model_type).encoder.layer)
            layers.reverse()
            lr = args.learning_rate
            for layer in layers:
                lr *= args.layerwise_learning_rate_decay
                optimizer_grouped_parameters += [
                    {
                        "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": args.weight_decay,
                        "lr": lr,
                    },
                    {
                        "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
        else:
            raise NotImplementedError
    return optimizer_grouped_parameters


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        with open(f"{args.output_dir}/raw_log.txt", "w") as f:
            pass  # create a new file

    if args.train_batch_size == 0:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_task_names = (args.task_name,)
    eval_datasets = [load_and_cache_examples(args, task, tokenizer, evaluate=True) for task in eval_task_names]
    if args.test_val_split:
        assert len(eval_datasets) == 1
        val_test_indices = []
        for i, eval_dataset in enumerate(eval_datasets):
            class2idx = defaultdict(list)
            for i, sample in enumerate(eval_dataset):
                class2idx[sample[-1].item()].append(i)
            val_indices = []
            test_indices = []
            for class_num, indices in class2idx.items():
                state = np.random.RandomState(1)
                state.shuffle(indices)
                class_val_indices, class_test_indices = indices[: len(indices) // 2], indices[len(indices) // 2 :]
                val_indices += class_val_indices
                test_indices += class_test_indices
            val_indices = torch.tensor(val_indices).long()
            test_indices = torch.tensor(test_indices).long()
            val_test_indices.append((val_indices, test_indices))
            eval_dataset.tensors = [t[val_indices] for t in eval_dataset.tensors]

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    assert args.logging_steps == 0 or args.num_loggings == 0, "Can only use 1 logging option"
    if args.logging_steps == 0:
        assert args.num_loggings > 0
        args.logging_steps = t_total // args.num_loggings

    if args.warmup_ratio > 0:
        assert args.warmup_steps == 0
        args.warmup_steps = int(args.warmup_ratio * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)

    if args.use_torch_adamw:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            correct_bias=not args.use_bertadam,
        )

    optimizer = PriorWD(optimizer, use_prior_wd=args.prior_weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info(
            "  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch,
        )

    tr_loss, logging_loss = 0.0, 0.0
    best_val_acc = -100.0
    best_model = None
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args.seed)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type not in {"distilbert", "bart"}:
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and (
                    (args.logging_steps > 0 and global_step % args.logging_steps == 0) or (global_step == t_total)
                ):
                    logs = {}
                    if args.local_rank == -1:
                        results = evaluate(args, model, tokenizer, eval_datasets=eval_datasets)
                        for key, value in results.items():
                            eval_key = "val_{}".format(key)
                            logs[eval_key] = value

                    if args.local_rank in [-1, 0] and args.save_best and logs["val_acc"] > best_val_acc:
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(
                            optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    if "val_acc" in logs:
                        if logs["val_acc"] > best_val_acc:
                            best_val_acc = logs["val_acc"]
                            best_model = {k: v.cpu().detach() for k, v in model.state_dict().items()}
                        logs["best_val_acc"] = best_val_acc
                    elif "val_mcc" in logs:
                        if logs["val_mcc"] > best_val_acc:
                            best_val_acc = logs["val_mcc"]
                            best_model = {k: v.cpu().detach() for k, v in model.state_dict().items()}
                        logs["best_val_mcc"] = best_val_acc
                    elif "val_spearmanr":
                        if logs["val_spearmanr"] > best_val_acc:
                            best_val_acc = logs["val_spearmanr"]
                            best_model = {k: v.cpu().detach() for k, v in model.state_dict().items()}
                        logs["best_val_spearmanr"] = best_val_acc
                    else:
                        raise ValueError(f"logs:{logs}")

                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar

                    if args.logging_steps > 0:
                        if global_step % args.logging_steps == 0:
                            loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        else:
                            loss_scalar = (tr_loss - logging_loss) / (global_step % args.logging_steps)
                    else:
                        loss_scalar = (tr_loss - logging_loss) / global_step
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    logs["step"] = global_step
                    with open(f"{args.output_dir}/raw_log.txt", "a") as f:
                        if os.stat(f"{args.output_dir}/raw_log.txt").st_size == 0:
                            for k in logs:
                                f.write(f"{k},")
                            f.write("\n")
                        for v in logs.values():
                            f.write(f"{v:.6f},")
                        f.write("\n")

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-last".format(global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    args.resplit_val = 0  # test on the original test_set
    eval_task_names = (args.task_name,)

    # test the last checkpoint on the second half
    eval_datasets = [load_and_cache_examples(args, task, tokenizer, evaluate=True) for task in eval_task_names]
    if args.test_val_split:
        for i, eval_dataset in enumerate(eval_datasets):
            test_indices = val_test_indices[i][1]
            eval_dataset.tensors = [t[test_indices] for t in eval_dataset.tensors]

    result = evaluate(args, model, tokenizer, eval_datasets=eval_datasets)
    result["step"] = t_total
    # overwriting validation results
    with open(f"{args.output_dir}/test_last_log.txt", "w") as f:
        f.write(",".join(["test_" + k for k in result.keys()]) + "\n")
        f.write(",".join([f"{v:.4f}" for v in result.values()]))

    if best_model is not None:
        model.load_state_dict(best_model)

    # test on the second half
    eval_datasets = [load_and_cache_examples(args, task, tokenizer, evaluate=True) for task in eval_task_names]
    if args.test_val_split:
        for i, eval_dataset in enumerate(eval_datasets):
            test_indices = val_test_indices[i][1]
            eval_dataset.tensors = [t[test_indices] for t in eval_dataset.tensors]

    result = evaluate(args, model, tokenizer, eval_datasets=eval_datasets)
    result["step"] = t_total
    # overwriting validation results
    with open(f"{args.output_dir}/test_best_log.txt", "w") as f:
        f.write(",".join(["test_" + k for k in result.keys()]) + "\n")
        f.write(",".join([f"{v:.4f}" for v in result.values()]))

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", eval_datasets=None):
    eval_task_names = [args.task_name]
    eval_outputs_dirs = [args.output_dir]

    results = {}
    for i, (eval_task, eval_output_dir) in enumerate(zip(eval_task_names, eval_outputs_dirs)):
        if eval_datasets is None:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        elif isinstance(eval_datasets, list):
            eval_dataset = eval_datasets[i]
        else:
            raise ValueError("Wrong Pre-fetched Eval Set")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if args.model_type not in {"distilbert", "bart"}:
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

    results["loss"] = eval_loss

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if (evaluate and args.resplit_val <= 0) else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if args.downsample_trainset > 0 and not evaluate:
        assert (args.downsample_trainset + args.resplit_val) <= len(features)

    if args.downsample_trainset > 0 or args.resplit_val > 0:
        set_seed(0)  # use the same seed for downsample
        if output_mode == "classification":
            label_to_idx = defaultdict(list)
            for i, f in enumerate(features):
                label_to_idx[f.label].append(i)

            samples_per_class = args.resplit_val if evaluate else args.downsample_trainset
            samples_per_class = samples_per_class // len(label_to_idx)

            for k in label_to_idx:
                label_to_idx[k] = np.array(label_to_idx[k])
                np.random.shuffle(label_to_idx[k])
                if evaluate:
                    if args.resplit_val > 0:
                        label_to_idx[k] = label_to_idx[k][-samples_per_class:]
                    else:
                        pass
                else:
                    if args.resplit_val > 0 and args.downsample_trainset <= 0:
                        samples_per_class = len(label_to_idx[k]) - args.resplit_val // len(label_to_idx)
                    label_to_idx[k] = label_to_idx[k][:samples_per_class]

            sampled_idx = np.concatenate(list(label_to_idx.values()))
        else:
            if args.downsample_trainset > 0:
                sampled_idx = torch.randperm(len(features))[: args.downsample_trainset]
            else:
                raise NotImplementedError
        set_seed(args.seed)
        features = [features[i] for i in sampled_idx]

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = get_parser()
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args.seed)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    num_labels_old = AutoConfig.from_pretrained(args.model_name_or_path).num_labels
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels_old,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.model_type == "electra":
        model = ElectraForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    if num_labels != num_labels_old:
        config.num_labels = num_labels
        model.num_labels = num_labels
        if args.model_type in ["roberta", "bert", "electra"]:
            from transformers.modeling_roberta import RobertaClassificationHead

            model.classifier = (
                RobertaClassificationHead(config)
                if args.model_type == "roberta"
                else nn.Linear(config.hidden_size, config.num_labels)
            )
            for module in model.classifier.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
        elif args.model_type == "bart":
            from transformers.modeling_bart import BartClassificationHead

            model.classification_head = BartClassificationHead(
                config.d_model, config.d_model, config.num_labels, config.classif_dropout,
            )
            model.model._init_weights(model.classification_head.dense)
            model.model._init_weights(model.classification_head.out_proj)
        elif args.model_type == "xlnet":
            model.logits_proj = nn.Linear(config.d_model, config.num_labels)
            model.transformer._init_weights(model.logits_proj)
        else:
            raise NotImplementedError

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.reinit_pooler:
        if args.model_type in ["bert", "roberta"]:
            encoder_temp = getattr(model, args.model_type)
            encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
            encoder_temp.pooler.dense.bias.data.zero_()
            for p in encoder_temp.pooler.parameters():
                p.requires_grad = True
        elif args.model_type in ["xlnet", "bart", "electra"]:
            raise ValueError(f"{args.model_type} does not have a pooler at the end")
        else:
            raise NotImplementedError

    if args.reinit_layers > 0:
        if args.model_type in ["bert", "roberta", "electra"]:
            assert args.reinit_pooler or args.model_type == "electra"
            from transformers.modeling_bert import BertLayerNorm

            encoder_temp = getattr(model, args.model_type)
            for layer in encoder_temp.encoder.layer[-args.reinit_layers :]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        # Slightly different from the TF version which uses truncated_normal for initialization
                        # cf https://github.com/pytorch/pytorch/pull/5617
                        module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
                    elif isinstance(module, BertLayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()
        elif args.model_type == "xlnet":
            from transformers.modeling_xlnet import XLNetLayerNorm, XLNetRelativeAttention

            for layer in model.transformer.layer[-args.reinit_layers :]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        # Slightly different from the TF version which uses truncated_normal for initialization
                        # cf https://github.com/pytorch/pytorch/pull/5617
                        module.weight.data.normal_(mean=0.0, std=model.transformer.config.initializer_range)
                        if isinstance(module, nn.Linear) and module.bias is not None:
                            module.bias.data.zero_()
                    elif isinstance(module, XLNetLayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    elif isinstance(module, XLNetRelativeAttention):
                        for param in [
                            module.q,
                            module.k,
                            module.v,
                            module.o,
                            module.r,
                            module.r_r_bias,
                            module.r_s_bias,
                            module.r_w_bias,
                            module.seg_embed,
                        ]:
                            param.data.normal_(mean=0.0, std=model.transformer.config.initializer_range)
        elif args.model_type == "bart":
            for layer in model.model.decoder.layers[-args.reinit_layers :]:
                for module in layer.modules():
                    model.model._init_weights(module)

        else:
            raise NotImplementedError

    if args.mixout > 0:
        from mixout import MixLinear

        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features, module.out_features, bias, target_state_dict["weight"], args.mixout
                    )
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
    print(model)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == "__main__":
    main()
