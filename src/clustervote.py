import random
from itertools import accumulate
from collections import deque, defaultdict, namedtuple, Counter
from itertools import combinations
from nltk.tokenize import sent_tokenize
import numpy as np
from .utils import flatten, lens_to_slices
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from kneed import KneeLocator


def approximate_eps(dist_matrix, min_cluster_size=2, step_offset=0, plot_steps=False, smoothing=False, center_radius=1):
    center_radius = np.clip(center_radius, 0, 1)
    min_cluster_size = min(min_cluster_size, dist_matrix.shape[0]-1)

    # Get min_cluster_size nearest points in space for each point
    neighbors = NearestNeighbors(
        n_neighbors=min_cluster_size, metric="precomputed")
    neighbors_fit = neighbors.fit(dist_matrix)
    distances, indices = neighbors_fit.kneighbors()

    # Sort to represent neighbour distance quantiles
    pdistances = np.sort(distances, axis=0)
    # Get distances for potential bordering points
    pdistances = pdistances[:, min_cluster_size-1]

    # If texts are identical then minimal eps would suffice
    # maybe max?
    if np.mean(pdistances) < 1e-6:
        return 1e-2

    # center the series
    pfixed = (pdistances-pdistances.mean())

    # find value jumps
    pfixed_steps = -np.cumsum(pfixed)
    center_indx = np.argmax(pfixed_steps)

    # smooth spikes
    if smoothing:
        pfixed = savgol_filter(pfixed, 51, 10)

    # include points within center_radius
    thr = int(np.ceil(center_indx*(1-center_radius)))

    # limit undersampling to 3%
    lim = center_indx+int(len(pfixed)*0.03)

    data_rank = len(np.unique(pdistances[thr:lim]))
    # Cant calculate elbow with less than 3 points
    if len(pdistances[thr:lim]) < 3 or data_rank < 3:
        return 1-center_radius

    # Degree should not be more than the rank of the data (number of unique values)
    poly_degree = max(min(data_rank-1, 7), 1)
    kneedle = KneeLocator(np.arange(len(pdistances))[thr:lim], pdistances[thr:lim], curve="convex", direction="decreasing", S=1,
                          interp_method="polynomial", polynomial_degree=poly_degree)

    # sometimes there is no elbow
    if not kneedle.elbow:
        kneeidx = thr
    else:
        kneeidx = kneedle.elbow

    if plot_steps:
        q_rounded = np.round(pfixed, 3)
        start_point = np.where(q_rounded >= np.quantile(q_rounded, 0.05))[0][0]
        print("Poly degree: {}".format(poly_degree))
        print("Indx: {}, Value: {:.04f}".format(
            kneeidx, pdistances[kneeidx]))
        norm_coef = pfixed.max()/pfixed_steps.max()
        plt.plot(pfixed)
        plt.plot(pfixed_steps*norm_coef)
        # new elbow formula
        plt.plot((kneeidx, kneeidx),
                 (pfixed_steps[kneeidx]*norm_coef, pfixed[kneeidx]), 'g')
        # 5% quantile
        plt.plot((start_point, start_point),
                 (pfixed_steps[start_point]*norm_coef, pfixed[start_point]), 'b')
        print(kneedle.all_elbows, pdistances[kneeidx])
    return max(pdistances[kneeidx+step_offset], 1e-2)


def build_textdistmat(text_sents, vectorizer_func, inf_value=1e10, metric='cosine'):
    textlens = [len(text) for text in text_sents]
    sentvects = vectorizer_func(list(flatten(text_sents))).astype(np.float64)
    sentdists = pairwise_distances(sentvects, metric=metric)
    text_ranges = lens_to_slices(textlens)

    for tr in text_ranges:
        text_range = tr
        sentdists[text_range, text_range] = inf_value
    np.fill_diagonal(sentdists, 0)
    return sentdists, sentvects, text_ranges


def sep_sents(text, min_sent_len=1):
    return [sent for par in text.split("\n") for sent in sent_tokenize(par) if len(sent.split()) >= min_sent_len]


def get_labelpower(doc_labels):
    label_freqs = Counter(flatten(doc_labels))
    document_label_freqs = [Counter(doc) for doc in doc_labels]
    sent_labelpower = [[(label_freqs[lab]-doclabfreq[lab]) if lab >= 0 else 0 for lab in doc]
                       for doclabfreq, doc in zip(document_label_freqs, doc_labels)]
    return sent_labelpower


def pair_sentences(cluster_senttexts, vectorizer_function, mean_distance_limit=0.2,
                   verbose=False, starting_radius=0.5, metric='cosine', agg_type="mean"):
    """
    texts : list[str]
        Texts to clusterize
    vectorizer_function : callable(str)
        Function for sentence vectorization
    mean_distance_limit : float, default 0.2
        Controls the quality of DBSCAN clusterization, the lower value corresponds to more conservative clusters
    verbose : bool, default False
        Toggles parameter search report
    """

    agg_funcs = {
        "mean": np.mean,
        "max": np.max
    }
    agg_func = agg_funcs[agg_type]

    cluster_sentdists, cluster_sentvects, cluster_textranges = build_textdistmat(
        cluster_senttexts, vectorizer_function, metric=metric)

    back_search = False
    last_checked = None
    radius_queue = deque(np.hstack((np.arange(starting_radius, 0.91, 0.05),
                                    np.arange(0.9, 1.01, 0.01))))
    iternum = -1
    # rad range is the maximal distance from true center to consider point as elbow/knee
    # This is linear
    # TODO implement bin search
    while radius_queue:
        iternum += 1
        rad = radius_queue.popleft()
        eps = approximate_eps(cluster_sentdists, min_cluster_size=2, step_offset=0,
                              center_radius=rad, smoothing=False, plot_steps=verbose)
        sent_clusterizer = DBSCAN(eps=eps, min_samples=2, metric="precomputed")

        if verbose:
            print("Iteration {}:\nCalculating clusters for eps {:.03f} and center_radius {:.02f}".format(
                iternum, eps, rad))
            print("."*10)

        sent_clusterizer.fit(cluster_sentdists)
        labels = sent_clusterizer.labels_

        cluster_sentlabels = [labels[seg].tolist()
                              for seg in cluster_textranges]
        cluster_sentpower = get_labelpower(cluster_sentlabels)

        # Inverted label-sentence clusters
        clusters = defaultdict(list)
        for idx, lab in enumerate(labels):
            clusters[lab].append(idx)

        centroids = {lab: np.asarray(
            np.mean(cluster_sentvects[clusters[lab]], axis=0)) for lab in clusters.keys()}

        # measure distance to centroids and group by texts
        # TODO vectorize generator
        sent_clust_dists = np.array([paired_distances(centroids[lab].reshape(
            1, -1), s.reshape(1, -1), metric='cosine')[0] for s, l in zip(cluster_sentvects, labels)])
        cluster_dists = [sent_clust_dists[seg] for seg in cluster_textranges]

        # Get cluster-wise pairwise sentence distances
        # Should drop -1 label since it can't have pairs
        valid_cluster_labels = sorted([k for k in clusters.keys() if k != -1])
        cl_sd = [cluster_sentdists[tuple(np.hsplit(np.array(
            list(combinations(clusters[lab], 2))), 2))] for lab in valid_cluster_labels]

        # Why sort?
        # calculate agg inner cluster distance
        cl_sd_sorted = sorted([(lab, agg_func(el[el < 1e5])) for el, lab in zip(
            cl_sd, valid_cluster_labels)], key=lambda x: -x[1])

        # in case we were very stict and no clusters were formed
        if not cl_sd_sorted:
            if not back_search:
                if verbose:
                    print("!!!Initiating back search!!!")
                radius_queue = deque(np.arange(rad-0.01, last_checked, -0.01))
                back_search = True
            continue

        # find max mean inner cluster distance
        max_dist_reached = max(cl_sd_sorted, key=lambda x: x[1])[1]
        # use smoothed distance comparison?
        if max_dist_reached < mean_distance_limit:  # *0.95:
            if verbose:
                print("OK\n")
            break
        else:
            last_checked = rad
            if verbose:
                print("Maximum distance {:.03f} is greater than mean_distance_limit {:.03f}\n".format(
                    max_dist_reached, mean_distance_limit))
    if verbose:
        print("Final maximum distance {:.03f}\n".format(
            max(cl_sd_sorted, key=lambda x: x[1])[1]))

    if not (all([np.allclose((len(t), len(l), len(p)), len(t)) for t, l, p in zip(cluster_senttexts, cluster_sentlabels, cluster_sentpower)])):
        print("Results incomplete, please check the behaviour of vectorizer_function")

    # To replace!!!!!
    res_tuple = namedtuple("Pairs", "senttexts sentlabels sentpower dists")
    return res_tuple(cluster_senttexts, cluster_sentlabels, cluster_sentpower, cluster_dists)


def find_identical(texts, vectorizer_func, eps=1e-5):
    textvects = vectorizer_func(texts).astype(np.float64)
    textdists = pairwise_distances(textvects, metric="euclidean")
    # can't drop self
    np.fill_diagonal(textdists, 1e10)
    to_remove = set(j for i, j in zip(*np.nonzero(np.tril(textdists < eps).T)))
    return to_remove


def flatids_to_nested(ids, nested_lens):
    ids = sorted(ids)
    id_boundaries = accumulate(nested_lens)
    last_boundary = 0
    cur_boundary = next(id_boundaries)
    nested_ids = [[]]
    for idx in ids:
        while idx >= cur_boundary:
            last_boundary = cur_boundary
            cur_boundary = next(id_boundaries)
            nested_ids.append([])
        nested_ids[-1].append(idx-last_boundary)
    return nested_ids


def filter_identical(text_sents, vectorizer_func):
    bad_ids = sorted(find_identical(flatten(text_sents), vectorizer_func))
    text_lens = (len(sents) for sents in text_sents)
    nested_ids = (set(nested)
                  for nested in flatids_to_nested(bad_ids, text_lens))
    return [[sent for sid, sent in enumerate(sents) if sid not in bad_nids]
            for sents, bad_nids in zip(text_sents, nested_ids)]


def cluster_vote(cluster_texts, paraphrase_vectorizer_fun, sentpair_starting_radius,
                 paraphrase_dist_limit, paraphrase_metric="cosine", agg_type="max", min_power_thr=0.4, verbose=False):

    # Clusterize sentences with paraphrase embeddings
    sb_res = pair_sentences(cluster_texts, paraphrase_vectorizer_fun, mean_distance_limit=paraphrase_dist_limit, verbose=verbose,
                            starting_radius=sentpair_starting_radius, metric=paraphrase_metric, agg_type=agg_type)

    full_senttexts = sb_res.senttexts
    full_labels = sb_res.sentlabels

    full_labelpower = get_labelpower(full_labels)
    full_dists = sb_res.dists

    return full_labels, full_senttexts, full_labelpower, full_dists


def doc_sentlabels_to_mat(doc_labels):
    sentlabels = np.array(list(flatten(doc_labels)))
    labelmat = np.zeros(sentlabels.shape*2)
    for s in range(len(sentlabels)):
        labelmat[s, :] = (sentlabels == sentlabels[s]
                          if sentlabels[s] >= 0 else 0)

    return labelmat


def labels_to_clusters(flat_labels, flat_powers):
    clusters = defaultdict(list)
    for l, (lab, pow) in enumerate(zip(flat_labels, flat_powers)):
        clusters[lab].append((l, pow))
    return clusters


def sample_from_clusters(clusters,  selection_threshold=0, seed=1234):
    sent_ids = []
    for cl_id in clusters.keys():
        if cl_id < 0:
            continue
        cluster = list((sid, pow)
                       for sid, pow in clusters[cl_id] if pow >= selection_threshold)
        if not cluster:
            continue
        random.seed(seed)
        sent_id, _ = random.choice(cluster)
        sent_ids.append(sent_id)
    return sent_ids
