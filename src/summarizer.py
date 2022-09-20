from dataclasses import dataclass
from sklearn.metrics import pairwise_distances
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from collections import Counter, defaultdict
from itertools import product
from scipy.sparse.csgraph import connected_components
from typing import Any, List, Tuple

from src.clustervote import cluster_vote, doc_sentlabels_to_mat
from .utils import flatten
import re
from scipy.linalg import block_diag


def get_graph_components(transition_matrix):
    _, labels = connected_components(transition_matrix)
    groups = [np.where(labels == tag)[0] for tag in np.unique(labels)]
    return groups


class Summarizer:
    def __init__(self) -> None:
        pass

    def summarize(self, texts, max_len=200):
        raise NotImplementedError


def order_tiebreaker(ranks):
    return [(rank, -r) for r, rank in enumerate(ranks)]


def random_tiebreaker(ranks, seed=1111):
    np.random.seed(seed)
    ties = np.random.random(len(ranks))
    return [(rank, r) for r, rank in zip(ties, ranks)]


tie_breaker_funcs = {
    "order": order_tiebreaker,
    "random": random_tiebreaker
}


@dataclass
class extractive_summary:
    text_units: Tuple[str]
    scores: Tuple[float]
    document_map: Tuple[int]


class ExtractiveSummarizer(Summarizer):
    def __init__(self, tokenizer=word_tokenize) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def rank_text_units(self, text_units: List[str], texts=None, **kwargs):
        raise NotImplementedError

    def split_text(self, text):
        raise NotImplementedError

    def tokenize(self, text):
        return self.tokenizer(text)

    def summarize(self, texts, max_len=None, cutoff_prob=0, tie_breaker="order", **kwargs):
        def custom_argsort(array):
            return sorted(range(len(array)), key=array.__getitem__)

        assert isinstance(texts, (list, tuple)) and all((isinstance(t, str)
                                                         for t in texts)), "texts must be list or tuple of str"

        # how to solve parargraph split problem?
        # straighforward: pass original texts to ranker
        # unit prefabrication is important for robustness
        texts_unitized = tuple(self.split_text(text) for text in texts)
        unit_textmap = tuple((t,)*len(unts)
                             for t, unts in enumerate(texts_unitized))
        unit_textmap = tuple(flatten(unit_textmap))
        units = tuple(flatten(texts_unitized))

        # from this point 100% same for all summarizers
        # pass original texts since some
        ranks = self.rank_text_units(units, texts, **kwargs)

        # ties will happen for similar sentences
        # so we keep additional value for argsort
        tied_ranks = tie_breaker_funcs[tie_breaker](ranks)

        summary = []
        budget = max_len
        if not budget:
            summary = range(len(tied_ranks))
        else:
            for rid in custom_argsort(tied_ranks)[::-1]:
                if ranks[rid] == 0:
                    break
                cand = units[rid]
                budget -= len(self.tokenize(cand))
                if budget >= 0:
                    summary.append(rid)
            summary = sorted(summary)
        return extractive_summary(*zip(*((units[s], ranks[s], unit_textmap[s]) for s in sorted(summary))))


class MarkovSummarizer(ExtractiveSummarizer):
    def _prepare_transition_matrix(self, transition_matrix):
        assert len(transition_matrix.shape) == 2 and np.equal(
            *transition_matrix.shape), "Transition matrix must be square"
        assert transition_matrix.sum() != 0, "Transition matrix must be non-zero"
        # Transition matrix must have row-wise sum equal to one
        norm = transition_matrix.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1
        transition_matrix = transition_matrix / norm
        return transition_matrix

    def safe_power_method(self, transition_matrix, eps=1e-5, max_it=1000):
        transition_matrix = self._prepare_transition_matrix(transition_matrix)
        ranking = np.zeros(transition_matrix.shape[0])
        # Markov chain process won't converge if matrix is reducible (has unreachable states)
        # So we cover this by processing connected components independently
        for comp in get_graph_components(transition_matrix):
            submatrix = transition_matrix[np.ix_(comp, comp)]
            ranking[comp] = self._power_method(
                submatrix, eps=eps, max_it=max_it)
        return ranking

    def _power_method(self, transition_matrix, eps=1e-5, max_it=1000):
        #transition_matrix = self._prepare_transition_matrix(transition_matrix)
        n = transition_matrix.shape[0]
        ranking = np.ones(n)/n
        prev_ranking = None
        for it in range(max_it):
            if prev_ranking is not None and np.allclose(ranking, prev_ranking, atol=eps):
                break
            prev_ranking = ranking
            ranking = transition_matrix.T.dot(prev_ranking)
        return ranking

    def build_transition_matrix(self, text_units):
        raise NotImplementedError


class MatrixBuilder:
    def __init__(self) -> None:
        return None

    def __call__(self, text_units, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class UniformBias(MatrixBuilder):
    def __call__(self, text_units, *args: Any, **kwds: Any) -> Any:
        text_units = list(flatten(text_units))
        return np.ones(len(text_units))/len(text_units)


class QueryBias(MatrixBuilder):
    def __init__(self, vectorizer_func) -> None:
        super().__init__()
        self.vectorizer = vectorizer_func

    def __call__(self, text_units, query, *args: Any, **kwds: Any) -> Any:
        if not isinstance(query, str):
            raise ValueError("query must be of type str")
        query = [query]
        text_units = list(flatten(text_units))
        bias = 1 - pairwise_distances(self.vectorizer(text_units),
                                      self.vectorizer(query), metric="cosine")
        return bias.reshape(-1)


class EmbeddingSimilarityTransitions(MatrixBuilder):
    def __init__(self, vectorizer_func) -> None:
        super().__init__()
        self.vectorizer = vectorizer_func

    def __call__(self, text_units, *args: Any, **kwds: Any) -> Any:
        text_units = list(flatten(text_units))
        return 1 - pairwise_distances(self.vectorizer(text_units), metric="cosine")


class ClusterVoteTransitions(MatrixBuilder):
    def __init__(self, tf_idf_vectorizer_function, paraphrase_vectorizer_fun, sentpair_starting_radius,
                 paraphrase_dist_limit) -> None:
        super().__init__()
        self.tfidf_vectorizer = tf_idf_vectorizer_function
        self.para_vectorizer = paraphrase_vectorizer_fun
        self.sentpair_rad = sentpair_starting_radius
        self.para_distlimit = paraphrase_dist_limit

    def __call__(self, text_units, *args: Any, **kwds: Any) -> Any:
        labels, _, _, _ = cluster_vote(text_units, self.tfidf_vectorizer,
                                       self.para_vectorizer, self.sentpair_rad, self.para_distlimit)
        assert len(labels) == len(text_units), "Lost {} text units".format(
            len(text_units) - len(labels))
        res = doc_sentlabels_to_mat(labels)
        return res


class SubtextTransitionBooster(MatrixBuilder):
    def __init__(self, subtext_delimiter) -> None:
        super().__init__()
        self.splitter = re.compile(
            r'(?={})'.format(re.escape(subtext_delimiter)))

    def __call__(self, text_units, source_texts, *args: Any, **kwds: Any) -> Any:
        splits = [spl for text in source_texts
                  for spl in self.splitter.split(text)]
        # trivial case - dont boost anything
        if len(splits) == 1:
            return np.zeros((len(text_units),)*2)

        # can we compress this mess?
        # if we were guranteed to have unit boundaries
        # we could just sliced source texts!
        splid = 0
        shift = 0
        unit_labels = [0 for i in range(len(splits)+1)]
        for tu in text_units:
            while splid < len(splits):
                position = splits[splid].find(tu, shift)
                if position < 0:
                    splid += 1
                    shift = 0
                else:
                    unit_labels[splid] += 1
                    shift = position + 1
                    break
            if splid == len(splits):
                break
        # used for padding
        remainder = len(text_units)-sum(unit_labels)

        res = block_diag(*[np.ones((label_freq,)*2)
                           for label_freq in unit_labels if label_freq > 0]+[np.zeros((remainder,)*2)])
        return res


def check_and_normalize_weights(weights):
    weights = np.array(weights)
    return weights / weights.sum()


class CompositeTextRank(MarkovSummarizer):
    def __init__(self, bias_builders, transition_builders, bias_weights=None, transition_weights=None, text_splitter=sent_tokenize, tokenizer=word_tokenize):
        super().__init__(tokenizer)
        self.bias_builders = bias_builders
        self.transition_builders = transition_builders
        self.text_splitter = text_splitter
        self.bias_weights = bias_weights
        self.transition_weights = transition_weights
        if self.bias_weights:
            self.bias_weights = check_and_normalize_weights(
                self.bias_weights)
        if self.transition_weights:
            self.transition_weights = check_and_normalize_weights(
                self.transition_weights)

    def split_text(self, text):
        return self.text_splitter(text)

    def rank_text_units(self, text_units, texts=None, damping_factor=0.85, max_it=1000, **kwargs):
        bias = (bias_builder(text_units, source_texts=texts, **kwargs)
                for bias_builder in self.bias_builders)
        bias = [b/b.sum() if b.sum() else b for b in bias]
        transitions = (transition_builder(text_units, source_texts=texts, **kwargs)
                       for transition_builder in self.transition_builders)
        transitions = [self._prepare_transition_matrix(t) for t in transitions]
        transition_matrix = (1-damping_factor) * \
            np.average(bias, axis=0, weights=self.bias_weights) + \
            damping_factor*np.average(transitions,
                                      axis=0, weights=self.transition_weights)

        return self.safe_power_method(transition_matrix, max_it=max_it)


class LexRank(MarkovSummarizer):
    def __init__(self, idf_dictionary=None, documents=None, tokenizer=word_tokenize) -> None:
        super().__init__(tokenizer)
        if not idf_dictionary and not documents:
            raise ValueError(
                "At least idf_dictionary or documents must be specified")
        self.idf_dict = idf_dictionary
        if not self.idf_dict:
            self.idf_dict = self.build_idf_dictionary(documents)

    def build_idf_dictionary(self, documents, default_score=0):
        doc_wise_vocab = Counter()
        for doc_count, doc in enumerate(documents):
            vocabulary = set((w for w in self.tokenize(doc)))
            doc_wise_vocab.update(vocabulary)
        idf_dict = defaultdict(lambda: default_score)
        idf_dict.update(((w, np.log(doc_count/v))
                        for w, v in doc_wise_vocab.items()))
        return idf_dict

    def _idf_modified_cosine(self, tf_sa, tf_sb):
        def calc_tfidf_norm(tf_a, tf_b, words):
            return sum((tf_a[w] * self.idf_dict[w] * tf_b[w] * self.idf_dict[w] for w in words))

        if tf_sa == tf_sb:
            return 1

        words_sa, words_sb = set(tf_sa.keys()), set(tf_sb.keys())

        similarity = calc_tfidf_norm(tf_sa, tf_sb, words_sa & words_sb) / np.sqrt(
            calc_tfidf_norm(tf_sa, tf_sa, words_sa) * calc_tfidf_norm(tf_sb, tf_sb, words_sb))

        return similarity

    def build_transition_matrix(self, text_units):
        term_frequencies = [Counter(self.tokenize(tu)) for tu in text_units]
        transition_mat = np.fromiter((self._idf_modified_cosine(tf_sa, tf_sb) for tf_sa, tf_sb in product(
            term_frequencies, term_frequencies)), dtype=float)
        transition_mat = transition_mat.reshape((len(term_frequencies),)*2)
        return transition_mat

    def quantize_edges(self, transition_matrix, threshold):
        if threshold is not None:
            quantized_mat = np.zeros_like(transition_matrix)
            quantized_mat[transition_matrix >= threshold] = 1
            return quantized_mat
        return transition_matrix

    def rank_text_units(self, text_units, threshold=None, damping_factor=0.1):
        assert damping_factor <= 1 and damping_factor >= 0, "Damping factor must be in range (0,1)"
        transition_matrix = self.build_transition_matrix(text_units)
        transition_matrix = self.quantize_edges(transition_matrix, threshold)
        transition_matrix = damping_factor * 1 / transition_matrix.shape[0] \
            + (1 - damping_factor) * transition_matrix
        ranking = self.safe_power_method(transition_matrix)
        return ranking
