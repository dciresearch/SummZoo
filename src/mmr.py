import numpy as np
from sklearn.metrics import pairwise_distances


def mmr_ranking(unit_scores, unit_embeddings, top_k=0.15, diversity=0.8, metric='cosine'):
    assert top_k > 0, "top_k must be positive"
    if top_k < 1:
        top_k = np.ceil(len(unit_embeddings)*top_k)
    inter_unit_distance = pairwise_distances(unit_embeddings, metric=metric)
    unit_scores = np.array(unit_scores)
    selected_idx = [np.argmax(unit_scores)]

    candidates_idx = [i for i in range(
        unit_embeddings.shape[0]) if i != selected_idx[0]]
    # Diversify other top-k
    for _ in range(min(top_k - 1, unit_embeddings.shape[0] - 1)):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        document_candidate_similarities = unit_scores[candidates_idx]
        inter_candidate_distances = np.min(
            inter_unit_distance[candidates_idx][:, selected_idx], axis=1
        )

        # Calculate MMR
        mmr = (1 - diversity) * document_candidate_similarities \
            + diversity * inter_candidate_distances
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        selected_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    # Extract and sort keywords in descending order of similarity
    selected_scores = [
        unit_scores[idx]
        for idx in selected_idx
    ]
    selected_order = np.argsort(selected_scores)[::-1]
    return [selected_idx[o] for o in selected_order], [selected_scores[o] for o in selected_order]
