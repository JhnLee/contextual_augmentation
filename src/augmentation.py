import argparse
import random
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    BertConfig,
    BertTokenizer,
)

from data_utils import TextDataset, LineByLineTextDataset
from model import BertForMlmWithClassification

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMlmWithClassification, BertTokenizer),
}


def load_and_cache_examples(args, tokenizer):
    file_path = args.data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path)
    else:
        return TextDataset(tokenizer, args, file_path=file_path)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(
    inputs: torch.Tensor, tokenizer: BertTokenizer, args
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return inputs, labels


def inference(args, infer_dataset, model: BertForMlmWithClassification, tokenizer: BertTokenizer):
    output_dir = args.output_dir

    if args.local_rank in [-1, 0]:
        os.makedirs(output_dir, exist_ok=True)

    args.infer_batch_size = args.per_gpu_infer_batch_size

    def collate(data: List[torch.Tensor]):
        sentences, labels = list(zip(*data))
        if tokenizer._pad_token is None:
            return pad_sequence(sentences, batch_first=True)
        return (
            pad_sequence(sentences, batch_first=True, padding_value=tokenizer.pad_token_id),
            torch.tensor(labels),
        )

    infer_sampler = SequentialSampler(infer_dataset)
    infer_dataloader = DataLoader(
        infer_dataset, sampler=infer_sampler, batch_size=args.infer_batch_size, collate_fn=collate
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running inference *****")
    logger.info("  Num examples = %d", len(infer_dataset))
    logger.info("  Batch size = %d", args.infer_batch_size)
    infer_loss = 0.0
    nb_infer_steps = 0
    model.eval()
    cls_preds = None

    for batch in tqdm(infer_dataloader, desc="Inference"):
        if nb_infer_steps > 100:
            break

        batch, class_labels = batch

        inputs, mask_labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        class_labels = class_labels.to(args.device)
        mask_labels = mask_labels.to(args.device) if args.mlm else None

        with torch.no_grad():
            loss, mlm_scores, cls_scores = model(
                input_ids=inputs, masked_lm_labels=mask_labels, class_labels=class_labels
            )
            infer_loss += loss.mean().item()
        nb_infer_steps += 1

        mlm_preds = torch.softmax(mlm_scores, dim=2).detach().cpu().numpy()

        is_masked = inputs == tokenizer.mask_token_id

        mlm_preds = mlm_preds[is_masked.cpu().numpy()]

        vocabs = list(tokenizer.vocab.keys())

        mlm_sorted = np.array(
            [[vocabs[pred] for pred in mask] for mask in (-mlm_preds).argsort()[:, :3]]
        )

        true_mask_labels = [vocabs[t] for t in mask_labels[0][is_masked.cpu().numpy()]]

        logger.info(tokenizer.convert_ids_to_tokens(inputs[0]))
        logger.info(class_labels)
        logger.info(mlm_sorted)
        logger.info(true_mask_labels)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        required=True,
        help="The input data file for augmentation.",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--args_path", type=str, required=True, help="Predetermined argument file path.",
    )
    parser.add_argument(
        "--pretrained_model_path", type=str, required=True, help="Pretrained model for inference.",
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Predetermined config path.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--per_gpu_infer_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for inference.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    args = parser.parse_args()
    pre_args = torch.load(args.args_path)
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
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
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.model_type = pre_args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[pre_args.model_type]
    config = config_class.from_pretrained(args.config_path,)
    tokenizer = tokenizer_class.from_pretrained(
        pre_args.model_name_or_path, do_lower_case=pre_args.do_lower_case,
    )
    model = model_class.from_pretrained(args.pretrained_model_path, config=config,)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(pre_args.device)

    logger.info("Inference parameters %s", args)

    # Inference
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    infer_dataset = load_and_cache_examples(args, tokenizer)

    if args.local_rank == 0:
        torch.distributed.barrier()

    inference(args, infer_dataset, model, tokenizer)


if __name__ == "__main__":
    main()

