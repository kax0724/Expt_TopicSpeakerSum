import os
import numpy as np
import pandas as pd
import json

def load_corpus(corpus_dir, data_name='train'):

    assert data_name in ['train', 'val', 'test']

    df = pd.read_table(os.path.join(corpus_dir, data_name+".tsv"), index_col=0)
    df = df.dropna(subset=['dialogue'])

    source_list = list(df['dialogue'].values)
    target_list = list(df['summary'].values)
    print("{:<5} source :".format(data_name),len(source_list),", target:",len(target_list))

    return source_list, target_list


    return train_source_list, train_target_list, val_source_list, val_target_list, test_source_list, test_target_list

def check_no_newline(text_list):
    for i, text in enumerate(text_list):
        assert "\r" not in text and "\n" not in text

def save_data(text_list, output_dir, data_name, mode):
    with open(os.path.join(output_dir, '{}.{}'.format(data_name, mode)), 'w') as f:
        f.write('\n'.join(text_list))

def newline_to_sep(text_list):
    """
    改行を[SEP]に変換
    """
    return [text.replace("\r\n"," [SEP] ").replace("\n"," [SEP] ").replace("\r"," [SEP] ") for text in text_list]

def dialogue_preprocess(text_list):
    """
    [SAYS]: ':' after Speaker Names,
    [EOU] : End of Utterance(Sentence),
    [EOT] : End of a Speaker's Talk,

    Example:
    John [SAYS] Hi, Daive! [EOU] How are you? [EOU] [EOT] Daive [SAYS] I'm good, thanks. ([EOU] [EOT])
    """

    # Delete ':' after Speaker Names and Add [SAYS] Tokens
    pp_text_list = []
    flag_full_name = False
    for text in text_list:
        word_list = []
        for word_i, w in enumerate(text.split(" ")):
            if flag_full_name and w.endswith(':'):
                word_list.append(w[:-1])
                word_list.append("[SAYS]")
                flag_full_name = False
                continue
            if "\r\n" in w or "\n" in w or "\r" in w or word_i==0:
                if w.endswith(':'):
                    word_list.append(w[:-1])
                    word_list.append("[SAYS]")
                else:
                    word_list.append(w)
                    flag_full_name = True
            else:
                word_list.append(w)
        pp_text_list.append(" ".join(word_list))
    
    assert len(text_list) == len(pp_text_list)

    # Delete Newlines
    pp_text_list = [text.replace("\r\n"," ").replace("\n"," ").replace("\r"," ") for text in pp_text_list]

    # Add [EOU] or [EOT] before Speaker Name
    pp2_text_list = []
    for sentence_i, text in enumerate(pp_text_list):
        w_prev = "[NULL]"
        speaker_prev = "[NULL]"
        word_list = text.split(" ")
        acc = 0
        for word_i, w in enumerate(text.split(" ")):
            if w == "[SAYS]":
                if speaker_prev != "[NULL]" and speaker_prev == w_prev:
                    word_list[word_i-acc] = "[EOU]"
                    word_list.pop(word_i-1-acc)
                    acc += 1
                elif speaker_prev != "[NULL]" and speaker_prev != w_prev:
                    if word_list[word_i - 2 - acc] and word_list[word_i - 2 - acc][-1].isalpha():
                        word_list[word_i - 2 - acc] = word_list[word_i - 2 - acc] + "."
                    word_list.insert(word_i - 1 - acc, "[EOT]")
                    acc -= 1
                    
                speaker_prev = w_prev
            w_prev = w
        pp2_text_list.append(" ".join(word_list))
    
    assert len(text_list) == len(pp2_text_list)

    # Add [EOU] Tokens at End of Utterance(Sentence)
    pp2_text_list = [
        text.replace(". ",". [EOU] ")
            .replace("! ","! [EOU] ")
            .replace("? ","? [EOU] ") for text in pp2_text_list
        ]
    
    # Replace Consecutive Spaces for a Space
    pp2_text_list = [
        text.replace("    "," ")
            .replace("   "," ")
            .replace("  "," ") for text in pp2_text_list
        ]
    
    # Add '.'(Period) and [EOU] Tokens at End of Utterance(Sentence)
    pp3_text_list = []
    for sent_i, text in enumerate(pp2_text_list):
        w_prev = "[NULL]"
        w_prev_prev = "[NULL]"
        word_list = text.split(" ")
        acc = 0
        for word_i, w in enumerate(text.split(" ")):
            if w == "[EOT]" and w_prev == "[EOU]" and w_prev_prev[-1].isalpha():
                word_list[word_i-2+acc] = word_list[word_i-2+acc]+"."
            elif w == "[EOT]" and w_prev[-1].isalpha():
                word_list[word_i-1+acc] = word_list[word_i-1+acc]+"."
                word_list.insert(word_i+acc,"[EOU]")
                acc += 1
            elif w == "[EOT]" and w_prev != "[EOU]":
                word_list.insert(word_i+acc,"[EOU]")
                acc += 1
            w_prev_prev = w_prev
            w_prev = w
        pp3_text_list.append(' '.join(word_list))
    
    assert len(text_list) == len(pp3_text_list)

    # Replace Consecutive [EOU] Tokens for a [EOU] Token
    pp4_text_list = []
    for sent_i, text in enumerate(pp3_text_list):
        w_prev = "[NULL]"
        word_list = text.split(" ")
        acc = 0
        for word_i, w in enumerate(text.split(" ")):
            if w == "[EOU]" and w_prev == "[EOU]":
                word_list.pop(word_i+acc)
                acc -= 1
            w_prev = w
        pp4_text_list.append(' '.join(word_list))

    assert len(text_list) == len(pp4_text_list)

    # Check No Consecutive [EOU] Tokens
    for text in pp4_text_list:
        w_prev = "[NULL]"
        for w in text.split(" "):
            assert w != "[EOU]" or w_prev != "[EOU]"
            w_prev = w

    # Fix a Bit
    pp4_text_list = [text.replace(":D.", ":D") for text in pp4_text_list]

    # Add [EOU] and [EOT] Tokens at the End of Dialogues.
    pp4_text_list = [text+" [EOU] [EOT]" for text in pp4_text_list]

    return pp4_text_list

def preprocess(corpus_dir, output_dir, dialogue=False, del_eot=False):
    print("loading corpus...")
    train_source_list, train_target_list = load_corpus(corpus_dir, data_name='train')
    val_source_list, val_target_list = load_corpus(corpus_dir, data_name='val')
    test_source_list, test_target_list = load_corpus(corpus_dir, data_name='test')

    data_list = [
        train_source_list, train_target_list,
        val_source_list, val_target_list,
        test_source_list, test_target_list,
    ]
    
    print("preprocessing...")
    pp_data_list = []
    for i, text_list in enumerate(data_list):
        # print(i)
        if dialogue:
            pp_data_list.append(dialogue_preprocess(text_list))
        else:
            pp_data_list.append(newline_to_sep(text_list))

    print("checking no newline...")
    for i, text_list in enumerate(pp_data_list):
        check_no_newline(text_list)
    
    if del_eot:
        print("deleting [EOT]...")
        pp2_data_list = []
        for i, text_list in enumerate(pp_data_list):
            pp2_data_list.append([text.replace(" [EOT]","") for text in text_list])
    else:
        pp2_data_list = pp_data_list

    print("saving...")
    os.makedirs(output_dir, exist_ok=True)
    data_names = ['train', 'val', 'test']
    for i, text_list in enumerate(pp2_data_list):
        data_name = data_names[int(i/2)]
        mode = 'source' if i%2==0 else 'target'
        save_data(text_list, output_dir, data_name, mode)
    
    print("done!")

if __name__ == '__main__':
    corpus_dir = "/home/naraki/dialogsum/corpus"
    output_dir = "/home/naraki/dialogsum/samsum_dataset3"

    preprocess(corpus_dir, output_dir, dialogue=True, del_eot=True)