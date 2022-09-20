from .utils import load_obj
from transformers import AutoTokenizer
import fasttext
import re
from nltk.corpus import stopwords
from .hf_senttransformer import HuggingfaceSentTransformer

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


def get_supported_languages():
    return lang_para_vectorizers.keys() & lang_tf_idf_vectorizers.keys()


def separate_symbols(text):
    res = text
    res = re.sub(r"([^\w\s]+)", r" \1 ", res).strip()
    res = re.sub(r" +", r" ", res)
    return res


def prepare_text(text):
    res = text
    res = separate_symbols(res)
    # res = " ".join([tok for tok in res.split(
    #     " ") if tok.lower() not in ignored_wds])
    return res


class FeatureTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        res = prepare_text(text)
        res = [t for t in self.tokenizer.tokenize(
            res, add_special_tokens=False) if t != self.tokenizer.unk_token]
        return res


class TfIdfBPEVectorizer:
    def __init__(self, lang) -> None:
        path, model_name = lang_tf_idf_vectorizers[lang]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vectorizer = load_obj(path)
        self.vectorizer.tokenizer = FeatureTokenizer(tokenizer)
        self.vectorizer.smooth_idf = True

    def __call__(self, text):
        return self.vectorizer.transform(text)


class ParaVectorizer(HuggingfaceSentTransformer):
    def __init__(self, model_name_or_path=None, lang=None, pooling="mean", device=None, max_seq_length=None):
        assert model_name_or_path or lang, "Must specify lang of model_name_or_path"
        assert lang in lang_para_vectorizers, \
            "Default paraphrase vectorizer is not configured for language {}".format(
                lang)
        if model_name_or_path is None:
            model_name_or_path = lang_para_vectorizers[lang]
        super().__init__(model_name_or_path, pooling, device, max_seq_length)

    def __call__(self, text, batch_size=32):
        return self.encode(text)


class LanguageDetector:
    def __init__(self, model_path=DEFAULT_LANGUAGE_DETECTION_MODEL_PATH) -> None:
        self.model = fasttext.load_model(model_path)
        self._digital_words = re.compile(r"\b\w*\d+\w*\b")
        self._spaces = re.compile(r"\s+")

    def _normalize_text(self, text):
        text_norm = self._digital_words.sub("", text)
        text_norm = self._spaces.sub(" ", text_norm)
        return text_norm

    def detect(self, texts):
        texts = [re.sub(r"([^\.\!\?\n])\n", r"\1.\n", t) for t in texts]
        if isinstance(texts, str):
            texts = [texts]
        normalized_texts = [self._normalize_text(text) for text in texts]
        prediction = self.model.predict(normalized_texts)
        labels = [label[0].rsplit("__")[-1] for label in prediction[0]]
        probs = [prob[0] for prob in prediction[1]]
        return labels, probs
