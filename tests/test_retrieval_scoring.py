from app.retrieval.pipeline import RetrievalService


def test_distance_to_score_maps_cosine_distance_to_similarity():
    assert RetrievalService._distance_to_score(0.0) == 1.0
    assert RetrievalService._distance_to_score(0.25) == 0.75
    assert RetrievalService._distance_to_score(1.2) == 0.0