import numpy as np
from sklearn.metrics import pairwise_distances


def mmr(doc_embedding, unit_embeddings, top_k=0.15, diversity=0.8, metric='cosine'):
    if top_k < 1 and top_k > 0:
        top_k = np.ceil(len(unit_embeddings)*top_k)
    unit_doc_similarity = pairwise_distances(
        unit_embeddings, doc_embedding, metric=metric)
    inter_unit_similarity = pairwise_distances(unit_embeddings, metric=metric)

    # Simple top-1
    selected_idx = [np.argmax(unit_doc_similarity)]
    candidates_idx = [i for i in range(
        unit_embeddings.shape[0]) if i != selected_idx[0]]

    # Diversify other top-k
    for _ in range(min(top_k - 1, unit_embeddings.shape[0] - 1)):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = unit_doc_similarity[candidates_idx, :]
        target_similarities = np.max(
            inter_unit_similarity[candidates_idx][:, selected_idx], axis=1
        )

        # Calculate MMR
        mmr = (1 - diversity) * candidate_similarities \
            - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        selected_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    # Extract and sort keywords in descending similarity
    selected_scores = [
        round(float(unit_doc_similarity.reshape(1, -1)[0][idx]), 4)
        for idx in selected_idx
    ]
    selected_order = np.argsort(selected_scores)[::-1]
    return [selected_idx[o] for o in selected_order], [selected_scores[o] for o in selected_order]
