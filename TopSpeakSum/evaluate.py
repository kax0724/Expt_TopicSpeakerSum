import sys

import os
import argparse
import string
import datetime
import tempfile
import json
import time
import shutil
from bs_pyrouge import Rouge155
import nltk
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
parser.add_argument("--generated", type=str, help="generated output file.")
parser.add_argument("--golden", type=str, help="Gold output file.")
parser.add_argument("--duplicate_rate", type=float, default=0.7,
                    help="If the duplicat rate (compared with history) is large, we can discard the current sentence.")
parser.add_argument("--trunc_len", type=int, default=0,
                    help="Truncate line by the maximum length.")


args = parser.parse_args()

fin = open(args.generated, 'r', encoding='utf-8')
fgolden = open(args.golden, 'r', encoding='utf-8')
dedup_rate = args.duplicate_rate
trunc_len = args.trunc_len

_tok_dict = {"(": "-LRB-", ")": "-RRB-",
             "[": "-LSB-", "]": "-RSB-",
             "{": "-LCB-", "}": "-RCB-"}


def _is_digit(w):
    for ch in w:
        if not (ch.isdigit() or ch == ','):
            return False
    return True


def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and \
                input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'" + input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(
                input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ',' + input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and \
                input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.' + input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[
            -1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[
            i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i + 3
            while k + 2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    text = ' '.join([x for x in output_tokens])
    fine_text = text.replace(' ##', '')
    # return " ".join(output_tokens)
    return fine_text


def remove_duplicate(l_list, duplicate_rate):
    tk_list = [l.lower().split() for l in l_list]
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set & history_set) / len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list


def test_rouge(cand, ref):
    temp_dir = tempfile.mkdtemp()
    candidates = cand
    references = ref
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )


def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter


def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)


generated_list = []
for line in fin:
    buf = []
    for sentence in nltk.sent_tokenize(line.strip()):
        sentence = fix_tokenization(sentence)
        if any(get_f1(sentence, s) > 1.0 for s in buf):
            continue
        s_len = len(sentence.split())
        if s_len <= 4:
            continue
        buf.append(sentence)
    if dedup_rate < 1:
        buf = remove_duplicate(buf, dedup_rate)
    if trunc_len:
        num_left = trunc_len
        trunc_list = []
        for bit in buf:
            tk_list = bit.split()
            n = min(len(tk_list), num_left)
            trunc_list.append(' '.join(tk_list[:n]))
            num_left -= n
            if num_left <= 0:
                break
    else:
        trunc_list = buf
    trunc_list = [item.replace('-LSB-', '') for item in trunc_list]
    generated_list.append("\n".join(trunc_list))

golden_list = []
for line in fgolden:
    line = line.strip().replace(" <S_SEP> ", '\n')
    golden_list.append(line)

scores = test_rouge(generated_list, golden_list)
print(rouge_results_to_str(scores))
