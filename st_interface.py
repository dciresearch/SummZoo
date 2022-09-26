import pandas as pd
import streamlit as st
from src.clustervote import cluster_vote
from src.utils import flatten
from src.summarizer import ClusterVoteTransitions, CompositeTextRank, EmbeddingSimilarityTransitions, QueryBias, UniformBias
from src.vectorizers import TfIdfBPEVectorizer, ParaVectorizer, LanguageDetector, get_supported_languages
from src.mmr import mmr
import razdel
import numpy as np
from collections import Counter


def ru_word_tokenize(text):
    return [tok.text for tok in razdel.tokenize(text)]


def ru_sent_tokenize(text, min_sent_len=3):
    pars = text.split('\n')
    return [sent.text for par in pars for sent in razdel.sentenize(par) if len(sent.text.split(" ")) >= min_sent_len]


@st.cache(allow_output_mutation=True)
def load_vectorizers(lang='ru'):
    return TfIdfBPEVectorizer(lang=lang), ParaVectorizer(lang=lang)


@st.cache(allow_output_mutation=True)
def load_language_detector():
    return LanguageDetector()


st.set_page_config(
    page_title="Summarizer", layout="wide"
)
c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.title("Summarizer")
    st.header("")

lang_detector = load_language_detector()

documents = []
files = st.file_uploader("Загрузка текстовых файлов",
                         accept_multiple_files=True)
if files:
    for uploaded_file in files:
        doc = uploaded_file.read().decode("utf-8")
        documents.append(doc)

    languages = lang_detector.detect(documents)[0]

    major_language = Counter(languages).most_common(1)[0][0]

    if major_language not in get_supported_languages():
        st.warning("Language {} is not supported".format(major_language))
        st.stop()

ModelType = st.selectbox(
    "Summarizer type",
    ["Composite TextRank", "MMR", "ClusterVote"],
    help="На данный момент доступны только экстрактивные подходы",
)

if not files:
    st.markdown("## Пожалуйста, загрузите файлы")
    st.stop()


def weight_slider(name, additional_info=""):
    weight = st.slider(
        "Вес {}".format(name),
        min_value=0.0,
        max_value=1.,
        value=0.5,
        help="Степень влияния {} на процесс суммаризации. {}".format(
            name, additional_info),
    )
    return weight


if ModelType == "ClusterVote" and len(documents) < 2:
    st.markdown("## Доступно только для 2 документов и более")
    st.stop()

ce, c1, ce, c2, ce = st.columns(
    [0.07, 2, 0.07, 5, 0.07])
with c1:
    with st.form(key="my_form"):
        if ModelType == "Composite TextRank":
            top_k = st.slider("Top-k", 1, 100, 5)
            query = st.text_input("Запрос", "")
            query_weight = weight_slider("запроса")
            clustervote_weight = weight_slider(
                "ClusterVote", "Метод ClusterVote усиливает вес схожих предложений в рамках одного кластера.")
            dist_vec_type = st.selectbox("Модель эмбеддингов для рассчета расстояний",
                                         ("TF-IDF", "BERT"))
            damping_factor = 1-st.slider("Коэффициент смягчения", 0.0, 1.0, 0.15,
                                         help="Коэффициент смягчения эффекта изолированных (непохожих) вершин")
            use_mmr = st.checkbox("Использовать алгоритм сэмплирования MMR?",
                                  help="Увеличивает разнообразие предложений")
        if ModelType == "MMR":
            top_k = st.slider("Top-k", 1, 100, 5)
            diversity = st.slider("Diversity", 0.0, 1.0, 0.8,
                                  help="Контролирует размер штрафа за отбор семантически похожих предложений.")
        if ModelType == "ClusterVote":
            selection_threshold = st.slider("Порог числа документов для отбора предложения", 1, len(
                documents), 1, help="Контролирует уровень детализации")
            para_limit = st.slider("Максимальное расстояние парафраз",
                                   0.0, 0.5, 0.3, help="Контролирует точность определения парафраз")
            dist_vec_type = st.selectbox("Модель эмбеддингов для рассчета расстояний",
                                         ("TF-IDF", "BERT"))
        submitted = st.form_submit_button("Submit")

if not submitted:
    st.stop()

tfidf_vec, para_vec = load_vectorizers(major_language)

full_text = "\n".join(documents)
sents = [ru_sent_tokenize(d) for d in documents]
out_sents = list(flatten(sents))

if len(documents) < 2:
    clustervote_weight = 0

if ModelType == "Composite TextRank":
    dist_vec = para_vec if dist_vec_type == "BERT" else tfidf_vec
    bias_builders = [
        UniformBias(),
        QueryBias(vectorizer_func=dist_vec),
    ]
    transition_builders = [
        EmbeddingSimilarityTransitions(vectorizer_func=dist_vec),
        ClusterVoteTransitions(tfidf_vec, para_vec, 0.5, 0.3)
    ]
    bias_weights = [1, 2*query_weight]
    transition_weights = [1, 2*clustervote_weight]
    summarizer = CompositeTextRank(bias_builders, transition_builders,
                                   bias_weights, transition_weights, text_splitter=ru_sent_tokenize, tokenizer=ru_word_tokenize)
    ranking = summarizer.rank_text_units(
        sents, documents, damping_factor=damping_factor, query=query)
    scores = ranking
    if not use_mmr:
        sent_ids = np.argsort(-scores)[:top_k]
    else:
        sent_ids = np.argsort(-scores)[:top_k*2]
        mmr_sents = [out_sents[s] for s in sent_ids]
        doc_emedding = dist_vec([full_text.replace('\n', ' ')]).reshape(1, -1)
        unit_embeds = dist_vec(mmr_sents)
        mmr_sent_ids, _ = mmr(doc_emedding, unit_embeds, top_k, 0.8)
        sent_ids = [sent_ids[i] for i in mmr_sent_ids]

if ModelType == "ClusterVote":
    dist_vec = para_vec if dist_vec_type == "BERT" else tfidf_vec
    labels, _, power, _ = cluster_vote(
        sents, tfidf_vec, dist_vec, 0.4, para_limit)
    sent_ids = [p for p, pow in enumerate(
        flatten(power)) if pow >= selection_threshold]
    scores = [pow/max(flatten(power)) for pow in flatten(power)]

if ModelType == "MMR":
    vec = para_vec
    doc_emedding = vec(full_text.replace('\n', ' ')).reshape(1, -1)
    unit_embeds = vec(out_sents)
    sent_ids, scores = mmr(doc_emedding, unit_embeds, top_k, diversity)
    scores = dict(zip(sent_ids, scores))

with c2:
    sent_ids = sorted(sent_ids)
    selected_sents = [out_sents[s] for s in sent_ids]
    selected_scores = [scores[s] for s in sent_ids]
    df = pd.DataFrame(list(zip(selected_sents, selected_scores)), columns=[
        "Извлеченные предложения", "Вероятность"])
    st.table(df)
