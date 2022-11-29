#!/usr/bin/env python

import os
import argparse
import datetime
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List
import pickle

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import cal_exact_rouge, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params
from finetune import SummarizationModule

logger = getLogger(__name__)


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_summaries_or_translations(
    examples: List[str],
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    prefix=None,
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)

    # loading hparams...
    with open(model_name.replace("best_tfmr","hparams.pkl"), 'rb') as f:
        hparams = pickle.load(f)
    
    model: SummarizationModule = SummarizationModule(hparams)

    if hparams.expand_vocab:
        special_tokens_dict = {'additional_special_tokens': ['[SAYS]','[EOU]','[EOT]']}
        num_added_toks = model.tokenizer.add_special_tokens(special_tokens_dict)
        model.model.resize_token_embeddings(len(model.tokenizer))

    if hparams.use_speaker_embeds or hparams.use_turn_embeds:
        from speaker_embed_encoder import BartEncoderWithSpeakerEmbedding
        model.model.model.encoder = BartEncoderWithSpeakerEmbedding(
            model.config,
            model.model.model.shared,
            ratio_to_token_embedding=hparams.ratio_to_token_embedding,
            speaker_embed_scale=hparams.speaker_embed_scale,
            use_turn_embeds=hparams.use_turn_embeds,
            partial_embed=hparams.partial_embed,
            )

    checkpoint = torch.load(model_name.replace("best_tfmr","val_avg_rouge2=29.0483-step_count=11.ckpt"))['state_dict']
    model.load_state_dict(checkpoint)
    model = model.to('cuda')

    start_time = time.time()
    # update config with task specific params
    use_task_specific_params(model.model, task)
    if prefix is None:
        prefix = prefix or getattr(model.model.config, "prefix", "") or ""
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        examples_chunk = [prefix + text for text in examples_chunk]
        batch = model.tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
        summaries = model.model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            num_beams=8,
            max_length=64,
            **generate_kwargs,
        )
        dec = model.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()
    fout.close()
    runtime = int(time.time() - start_time)  # seconds
    n_obs = len(examples)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))


def datetime_now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_generate(verbose=True):
    """

    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.

    Args:
        verbose (:obj:`bool`, `optional`, defaults to :obj:`True`): print results to stdout

    Returns:
        a tuple: ``(scores, params}``
        - ``scores``: a dict of scores data ``{'bleu': 39.6501, 'n_obs': 2000, 'runtime': 186, 'seconds_per_sample': 0.093}``
        - ``params``: a dict of custom params, e.g. ``{'num_beams': 5, 'length_penalty': 0.8}``
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("dataset_dir", type=str, help="like cnn_dm/test.source")
    # parser.add_argument("model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    # parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
    # parser.add_argument("save_path", type=str, help="where to save summaries")
    # parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    # parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dump-args", action="store_true", help="print the custom hparams with the results")
    parser.add_argument(
        "--info",
        nargs="?",
        type=str,
        const=datetime_now(),
        help="use in conjunction w/ --dump-args to print with the results whatever other info you'd like, e.g. lang=en-ru. If no value is passed, the current datetime string will be used.",
    )
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")

    args.model_name = os.path.join(args.model_dir, "best_tfmr")
    # args.input_path = os.path.join(args.dataset_dir, "test.source")
    # args.save_path = os.path.join(args.model_dir, "gen_summary.txt")
    # args.reference_path = os.path.join(args.dataset_dir, "test.target")
    # args.score_path = os.path.join(args.model_dir, "score.json")
    if args.val:
        args.input_path = os.path.join(args.dataset_dir, "val.source")
        args.reference_path = os.path.join(args.dataset_dir, "val.target")
        args.save_path = os.path.join(args.model_dir, "val_gen_summary.txt")
        args.score_path = os.path.join(args.model_dir, "val_score.json")
    else:
        args.input_path = os.path.join(args.dataset_dir, "test.source")
        args.reference_path = os.path.join(args.dataset_dir, "test.target")
        args.save_path = os.path.join(args.model_dir, "test_gen_summary.txt")
        args.score_path = os.path.join(args.model_dir, "test_score.json")

    
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]
    if args.n_obs > 0:
        examples = examples[: args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    runtime_metrics = generate_summaries_or_translations(
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        prefix=args.prefix,
        **parsed_args,
    )

    if args.reference_path is None:
        return {}

    # Compute scores
    score_fn = cal_exact_rouge if args.exact else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    scores.update(runtime_metrics)

    if args.dump_args:
        scores.update(parsed_args)
    if args.info:
        scores["info"] = args.info

    if verbose:
        print(scores)

    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w"))

    return scores


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME(./output/2020-MM-DD-HH-MM-SS/best_tfmr) $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    # Usage for Summarization
    # python run_eval.py output/2020-12-07-14-20-34 ../../samsum_dataset2
    run_generate(verbose=True)
