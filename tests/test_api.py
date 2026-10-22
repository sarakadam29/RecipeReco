import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main as api


@pytest.fixture
def client_with_dataset(tmp_path, monkeypatch):
    data_path = tmp_path / "recipes_clean.csv"
    recipes = pd.DataFrame(
        [
            {
                "id": 10,
                "name": "Tomato basil soup",
                "minutes": 25,
                "ingredients": '["tomato", "basil", "water"]',
                "description": "A simple tomato and basil soup.",
                "steps": '["Chop the tomatoes.", "Simmer with basil.", "Serve warm."]',
                "tags": "['indian']",
                "calories": 120.0,
                "total_fat_pdv": 4.0,
                "sugar_pdv": 5.0,
                "sodium_pdv": 6.0,
                "protein_pdv": 7.0,
                "saturated_fat_pdv": 2.0,
                "carbohydrates_pdv": 8.0,
                "cholesterol_pdv": None,
                "fiber_pdv": None,
            },
            {
                "id": 20,
                "name": "Tomato soup",
                "minutes": 15,
                "ingredients": '["tomato", "water"]',
                "description": None,
                "steps": "[]",
                "tags": "[]",
                "calories": 80.0,
                "total_fat_pdv": 1.0,
                "sugar_pdv": 2.0,
                "sodium_pdv": 3.0,
                "protein_pdv": 4.0,
                "saturated_fat_pdv": 0.0,
                "carbohydrates_pdv": 5.0,
                "cholesterol_pdv": None,
                "fiber_pdv": None,
            },
            {
                "id": 30,
                "name": "Basil soup",
                "minutes": 30,
                "ingredients": '["basil", "water"]',
                "description": None,
                "steps": "[]",
                "tags": "[]",
                "calories": 100.0,
                "total_fat_pdv": 1.0,
                "sugar_pdv": 2.0,
                "sodium_pdv": 3.0,
                "protein_pdv": 4.0,
                "saturated_fat_pdv": 0.0,
                "carbohydrates_pdv": 5.0,
                "cholesterol_pdv": None,
                "fiber_pdv": None,
            },
        ]
    )
    recipes.to_csv(data_path, index=False)
    monkeypatch.setattr(api, "DATASET_PATH", data_path)
    api._load_recommender.cache_clear()
    api._load_recipes.cache_clear()
    with TestClient(api.app) as client:
        yield client
    api._load_recommender.cache_clear()
    api._load_recipes.cache_clear()


def test_health_endpoint(client_with_dataset):
    response = client_with_dataset.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_returns_ranked_recipes_and_available_nutrition(client_with_dataset):
    response = client_with_dataset.post(
        "/recommend",
        json={
            "ingredients": [" TOMATO ", "basil"],
            "max_missing_ingredients": 1,
            "limit": 2,
            "flexible_matching": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    results = body["recommendations"]
    assert [recipe["recipe_id"] for recipe in results] == [10, 20]
    assert results[0]["matched_ingredients"] == ["tomato", "basil"]
    assert results[0]["missing_ingredients"] == []
    assert results[1]["matched_ingredients"] == ["tomato"]
    assert results[1]["missing_ingredients"] == ["basil"]
    assert results[0]["extra_ingredients"] == ["water"]
    assert results[0]["pantry_extras"] == ["water"]
    assert results[0]["additional_ingredients"] == []
    assert results[0]["match_status"] == "full_match"
    assert results[0]["dietary_classification"] == "vegan"
    assert results[0]["ingredients"] == ["tomato", "basil", "water"]
    assert results[0]["description"] == "A simple tomato and basil soup."
    assert results[0]["steps"] == [
        "Chop the tomatoes.",
        "Simmer with basil.",
        "Serve warm.",
    ]
    assert results[0]["cuisine_category"] == "Indian"
    assert results[0]["cuisine_classification_basis"] == "inferred_from_dataset_tags"
    assert results[0]["cooking_time_minutes"] == 25
    assert results[0]["nutrition"]["calories"] == 120
    assert results[0]["nutrition"]["total_fat_pdv"] == 4
    assert results[0]["nutrition"]["cholesterol_pdv"] is None
    assert results[0]["similarity_score"] > 0
    assert "does not specify whether nutrition values are per serving" in body["nutrition_basis"]
    assert "percentages of Daily Value" in body["nutrition_basis"]


def test_recommend_rejects_empty_or_malformed_requests(client_with_dataset):
    assert client_with_dataset.post("/recommend", json={"ingredients": []}).status_code == 422
    assert client_with_dataset.post(
        "/recommend", json={"ingredients": [" ", "\t"]}
    ).status_code == 422
    assert client_with_dataset.post(
        "/recommend", json={"ingredients": "tomato"}
    ).status_code == 422
    assert client_with_dataset.post(
        "/recommend", json={"ingredients": ["tomato"], "max_missing_ingredients": -1}
    ).status_code == 422
    assert client_with_dataset.post(
        "/recommend", json={"ingredients": ["tomato"], "limit": 0}
    ).status_code == 422
    assert client_with_dataset.post(
        "/recommend", json={"ingredients": ["tomato"], "limit": 101}
    ).status_code == 422


def test_recommend_uses_flexible_matching_by_default_and_accepts_missing_limit(
    client_with_dataset,
):
    request = {"ingredients": ["tomato", "basil", "onion", "pepper"]}

    default_response = client_with_dataset.post("/recommend", json=request)
    limited_response = client_with_dataset.post(
        "/recommend",
        json={
            **request,
            "max_missing_ingredients": 1,
            "flexible_matching": True,
        },
    )

    assert default_response.status_code == 200
    assert [
        item["recipe_id"]
        for item in default_response.json()["recommendations"]
    ] == [10, 20, 30]
    assert limited_response.status_code == 200
    assert limited_response.json()["recommendations"] == []


def test_recommend_supports_opt_in_strict_matching(client_with_dataset):
    strict = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["tomato", "basil"], "flexible_matching": False},
    )
    flexible = client_with_dataset.post(
        "/recommend",
        json={
            "ingredients": ["tomato", "basil"],
            "flexible_matching": True,
            "max_missing_ingredients": 1,
        },
    )

    assert [item["recipe_id"] for item in strict.json()["recommendations"]] == [10]
    partial = flexible.json()["recommendations"][1]
    assert partial["recipe_id"] == 20
    assert partial["match_status"] == "partial_match"
    assert partial["missing_ingredients"] == ["basil"]


def test_recommend_dietary_filters_and_rejects_unknown_diet(client_with_dataset):
    vegan = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["tomato", "basil"], "diet": "vegan"},
    )
    invalid = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["tomato"], "diet": "pescatarian"},
    )

    assert vegan.status_code == 200
    assert all(
        item["dietary_classification"] == "vegan"
        for item in vegan.json()["recommendations"]
    )
    assert invalid.status_code == 422


def test_recommend_can_filter_indian_and_allows_old_requests(client_with_dataset):
    old_request = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["tomato"], "limit": 3},
    )
    indian_request = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["tomato"], "cuisine": "indian"},
    )
    invalid_cuisine = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["tomato"], "cuisine": "mexican"},
    )

    assert old_request.status_code == 200
    assert old_request.json()["recommendations"]
    assert [row["recipe_id"] for row in indian_request.json()["recommendations"]] == [10]
    assert indian_request.json()["recommendations"][0]["cuisine_category"] == "Indian"
    assert invalid_cuisine.status_code == 422


def test_recommend_returns_empty_detail_fields_when_source_details_are_missing(
    client_with_dataset,
):
    response = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["tomato", "water"], "limit": 3},
    )

    assert response.status_code == 200
    second = next(
        row for row in response.json()["recommendations"] if row["recipe_id"] == 20
    )
    assert second["description"] is None
    assert second["steps"] == []


def test_recommend_explains_when_no_entered_ingredients_match(client_with_dataset):
    response = client_with_dataset.post(
        "/recommend",
        json={"ingredients": ["rice", "cabbage", "onion"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert "No relevant recipes matched any of your entered ingredients" in body["message"]


def test_recommend_returns_service_unavailable_when_dataset_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATASET_PATH", tmp_path / "missing.csv")
    api._load_recommender.cache_clear()
    api._load_recipes.cache_clear()
    try:
        with TestClient(api.app) as client:
            response = client.post("/recommend", json={"ingredients": ["tomato"]})
    finally:
        api._load_recommender.cache_clear()
        api._load_recipes.cache_clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "Processed recipe data is unavailable"


def test_recommend_reuses_cached_tfidf_index(client_with_dataset):
    loaded_recipes = api._load_recipes()
    first = api._load_recommender()
    client_with_dataset.post("/recommend", json={"ingredients": ["tomato"]})
    second = api._load_recommender()

    assert api._load_recipes() is loaded_recipes
    assert first is second
    assert api._load_recipes.cache_info().misses == 1
