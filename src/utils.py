from itertools import accumulate
from collections import Counter
import numpy as np
import json
import nltk
import pickle
from pathlib import Path


def save_obj(obj, fname, save_dir="./obj_cache/"):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / fname, "wb") as fOut:
        pickle.dump(obj, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as fIn:
        return pickle.load(fIn)


def prec_rec(count_a, count_b):
    at = sum(count_a.values())
    bt = sum(count_b.values())
    inter = sum((count_a & count_b).values())
    if inter == 0:
        return (0, 0)
    return (inter/at, inter/bt)


def ngrams(wds, n):
    return [" ".join(wds[i:i+n]) for i in range(0, len(wds)-n+1, 1)]


def rouge_ngrams(wds, n):
    return Counter(ngrams(wds, n))


def rouge_n(a, b, n=1):
    ac = rouge_ngrams(a, n)
    bc = rouge_ngrams(b, n)

    return prec_rec(ac, bc)


def rouge_l(a, b):
    def lcs(a, b):
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, ca in enumerate(a, 1):
            for j, cb in enumerate(b, 1):
                table[i][j] = (table[i - 1][j - 1] + 1 if ca == cb else
                               max(table[i][j - 1], table[i - 1][j]))
        return table[-1][-1]

    at = len(a)
    bt = len(b)

    inter = lcs(a, b)
    return (inter/at, inter/bt)


def rouge_score(a, b, rouge_type=1):
    if isinstance(rouge_type, int):
        return rouge_n(a, b, n=rouge_type)
    elif rouge_type.lower() == 'l':
        return rouge_l(a, b)
    else:
        raise Exception("rouge_type = {} is not defined".format(rouge_type))


def build_oracle(src, gold, rouge_type=1, mode="greedy", topk=5):
    """
    src=[[tokens for tokens in sent] for sent in doc]
    gold=[tokens for tokens in doc]
    """
    src_sents = src

    if mode == "greedy":
        gold_wds = gold
        scores = [rouge_score(s, gold, rouge_type)[1] for s in src_sents]
        ranked = sorted(zip(range(len(src_sents)), src_sents,
                        scores), key=lambda x: -x[-1])
        return sorted(ranked[:topk])


def match_gold_src(src, gld, topk=3, min_score=0, rouge_type="l"):
    src_sentences = src
    gold_sentences = gld

    candidates = []

    for g in gold_sentences:
        scores = [rouge_score(s, g, rouge_type=rouge_type)[1]
                  for s in src_sentences]
        topk = min(topk, len(scores))
        ranked = sorted(enumerate(scores), key=lambda x: -x[-1])[:topk]
        # Replace elements under the threshold by (-1,-1) duds
        ranked = [r if r[-1] >= min_score else (-1, -1) for r in ranked]

        candidates.append(ranked)

    selected = set()
    not_set = set(range(len(gold_sentences)))

    for level in range(topk):
        if not not_set:
            break
        local_candidates = [[cid, *c[level]]
                            for cid, c in enumerate(candidates)]
        local_candidates = sorted(local_candidates, key=lambda x: -x[-1])
        for lc in local_candidates:
            if (lc[0] in not_set) and (not lc[1] in selected):
                selected.add(lc[1])
                not_set.remove(lc[0])

    # Remove (-1,-1) dud from the results
    selected.discard(-1)
    return sorted(list(selected))


def tok_join(tokens, fix_punkt=False):
    import re
    res = " ".join(tokens).replace('’', "\'")
    res = re.sub("\s(\')\s([smtd]|re|ve)(?:\W|$)", r'\1\2 ', res)
    res = re.sub("\s(\'\w)", r'\1', res)
    res = re.sub("\s(n\'t)", r'\1', res)
    res = re.sub("\s(\'s)", r'\1', res)
    res = re.sub("(\s\w+\-)\s", r'\1', res)
    res = re.sub("\s(\-\w+\s)", r'\1', res)
    if fix_punkt:
        res = re.sub("\s([\.\,\!\?\;:])", r'\1', res)
    return res


def iterable(obj):
    try:
        iter(obj)
        if isinstance(obj, str):
            return False
        return True
    except:
        return False


def flatten(arr):
    if iterable(arr):
        for el in arr:
            yield from flatten(el)
    else:
        yield arr


def flatten_2d_map(obj):
    return tuple(zip(*[(el, oi) for oi, ocont in enumerate(obj) for el in ocont]))


def jdump(data, fp):
    json.dump(data, fp, ensure_ascii=False)
    fp.write("\n")


def subchain(index_lists):
    res = None
    for l in index_lists:
        if res is None:
            res = np.array(l)
        else:
            res = res[list(l)]
    return (res)


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)


def nltk_tokenize(text, drop_stopwords=True, language="english"):
    stopwords = []
    if drop_stopwords:
        stopwords = set(nltk.corpus.stopwords.words(language))
    return [tok for tok in nltk.word_tokenize(text) if tok.lower() not in stopwords]


def f1_score(precision, recall):
    return f_score(precision, recall, beta=1)


def f_score(precision, recall, beta=1):
    assert beta <= 2 and beta > 0, "beta must be in range (0,2)"
    denom = beta*beta*precision+recall
    if (denom) == 0:
        return 0
    return (1+beta*beta)*precision*recall/denom


def sum_rouge(doc, gold, weigths=[1, 1, 1], beta=1):
    rs = [f_score(*rouge_n(doc, gold, n=i), beta)*w for i,
          w in enumerate((weigths), start=1)]
    return sum(rs)


def calc_rouge(doc, gold, drop_stopwords=False, rouge_type=1):
    doc_tokenized = nltk_tokenize(doc.lower(), drop_stopwords=drop_stopwords)
    gold_tokenized = nltk_tokenize(gold.lower(), drop_stopwords=drop_stopwords)
    prec, rec = rouge_score(doc_tokenized, gold_tokenized, rouge_type)
    return {"precision": prec, "recall": rec, "f1": f1_score(prec, rec)}


def greedy_oracle(doc, gold, drop_stopwords=False, weight_scheme=[1, 1, 1], min_sent_length=5, max_its=100, beta=1):
    assert isinstance(doc, list) and all((isinstance(d, str)
                                          for d in doc)), "doc must be list of str"
    assert isinstance(gold, list) and all((isinstance(d, str)
                                           for d in gold)), "gold must be list of str"
    doc_tokenized = [nltk_tokenize(
        sent, drop_stopwords=drop_stopwords) for sent in doc]
    gold_tokenized = list(
        flatten([nltk_tokenize(sent, drop_stopwords=drop_stopwords) for sent in gold]))

    selected = []
    selected_ids = set()
    variants = set(range(len(doc_tokenized)))
    for i in range(max_its):
        purge = set()
        best_id = -1
        best_score = sum_rouge(selected, gold_tokenized,
                               weight_scheme, beta=beta)
        for s in variants:
            sent = doc_tokenized[s]
            if s in selected_ids or len(sent) < min_sent_length:
                purge.add(s)
                continue
            candidate_score = sum_rouge(
                selected+sent, gold_tokenized, weight_scheme, beta=beta)
            if candidate_score >= best_score:
                best_id = s
                best_score = candidate_score
        variants = variants.difference(purge)
        if best_id >= 0:
            variants.remove(best_id)
            selected_ids.add(best_id)
            selected.extend(doc_tokenized[best_id])
        else:
            return sorted(list(selected_ids)), best_score
    print("OUT OF ITS! RETURNING WHAT LEFT")
    return sorted(list(selected_ids)), best_score


def ends_to_segs(end_list):
    return [(start, end) for start, end in zip(accumulate([0]+end_list), accumulate(end_list))]


def lens_to_slices(len_list):
    return [slice(start, end) for start, end in zip(accumulate([0]+len_list), accumulate(len_list))]


def batch_generator(iterable, batch_size=10):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
