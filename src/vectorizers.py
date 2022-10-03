from .utils import load_obj
from transformers import AutoTokenizer
import fasttext
import re
from nltk.corpus import stopwords
from .hf_senttransformer import HuggingfaceSentTransformer
from typing import List, Set, Union

# in case if python version < 3.8.3
try:
    import pickle5 as pickle
except ImportError as e:
    pass


DEFAULT_LANGUAGE_DETECTION_MODEL_PATH = "./models/fasttext/lid.176.ftz"
ignored_wds = set(stopwords.words("english"))

lang_para_vectorizers = {
    "ru": './models/rucybert_70_torch',
    "en": "sentence-transformers/all-mpnet-base-v2",
}

lang_tf_idf_vectorizers = {
    "ru": ("./models/vectorizers/tfidf_ru/lenta_tokenizer_tfidf_ruRoberta-large.bin", "sberbank-ai/ruRoberta-large"),
    "en": ("./models/vectorizers/tfidf_en/allthenews_tokenizer_tfidf_bert-base-cased.bin", "bert-base-cased")
}

lang_aliases = {
    "ru": "russian",
    "en": "english"
}


def get_supported_languages():
    return lang_para_vectorizers.keys() & lang_tf_idf_vectorizers.keys()


def separate_symbols(text: str):
    res = text
    res = re.sub(r"([^\w\s]+)", r" \1 ", res).strip()
    res = re.sub(r" +", r" ", res)
    return res


def remove_symbols(text: str):
    res = text
    res = re.sub(r"([^\w\s]+)", r" ", res).strip()
    res = re.sub(r" +", r" ", res)
    return res


def fix_case(text: str):
    return text if not text.isupper() else text.lower()


def prepare_text(text, ignored_wds: Set[str]):
    res = fix_case(text)
    res = remove_symbols(res)
    res = " ".join((tok for tok in res.split(
        " ") if tok.lower() not in ignored_wds))
    return res


class FeatureTokenizer:
    def __init__(self, tokenizer, ignored_wds: Set[str] = None):
        self.tokenizer = tokenizer
        self.ignored_wds = ignored_wds if ignored_wds else []

    def __call__(self, text: str):
        res = prepare_text(text, self.ignored_wds)
        res = [t for t in self.tokenizer.tokenize(
            res, add_special_tokens=False) if t != self.tokenizer.unk_token]
        return res


class TfIdfBPEVectorizer:
    def __init__(self, lang: str) -> None:
        path, model_name = lang_tf_idf_vectorizers[lang]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vectorizer = load_obj(path)
        self.stopwords = set(stopwords.words(lang_aliases[lang]))
        self.vectorizer.tokenizer = FeatureTokenizer(tokenizer, self.stopwords)
        self.vectorizer.smooth_idf = True

    def __call__(self, texts: List[str]):
        return self.vectorizer.transform(texts)


class ParaVectorizer(HuggingfaceSentTransformer):
    def __init__(self, model_name_or_path: str = None, lang: str = None, pooling: str = "mean",
                 device: str = None, max_seq_length: int = None):
        assert model_name_or_path or lang, "Must specify lang of model_name_or_path"
        assert lang in lang_para_vectorizers, \
            "Default paraphrase vectorizer is not configured for language {}".format(
                lang)
        if model_name_or_path is None:
            model_name_or_path = lang_para_vectorizers[lang]
        super().__init__(model_name_or_path, pooling, device, max_seq_length)

    def __call__(self, texts: List[str], batch_size: int = 32):
        assert isinstance(texts, list), "texts must be list of strings"
        texts = [fix_case(t) for t in texts]
        return self.encode(texts, batch_size=batch_size)


class LanguageDetector:
    def __init__(self, model_path=DEFAULT_LANGUAGE_DETECTION_MODEL_PATH) -> None:
        self.model = fasttext.load_model(model_path)
        self._digital_words = re.compile(r"\b\w*\d+\w*\b")
        self._spaces = re.compile(r"\s+")

    def _normalize_text(self, text: str):
        text_norm = self._digital_words.sub("", text)
        text_norm = self._spaces.sub(" ", text_norm)
        return text_norm

    def detect(self, texts: Union[str, List[str]]):
        texts = [re.sub(r"([^\.\!\?\n])\n", r"\1.\n", t) for t in texts]
        if isinstance(texts, str):
            texts = [texts]
        normalized_texts = [self._normalize_text(text) for text in texts]
        prediction = self.model.predict(normalized_texts)
        labels = [label[0].rsplit("__")[-1] for label in prediction[0]]
        probs = [prob[0] for prob in prediction[1]]
        return labels, probs
