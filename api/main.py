import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.recipes import (
    NUTRITION_FIELDS,
    RecipeRecommender,
    parse_list_value,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "recipes_clean.csv"
RECIPE_DATA_COLUMNS = [
    "id",
    "name",
    "minutes",
    "ingredients",
    "description",
    "steps",
    "tags",
    *NUTRITION_FIELDS,
]
REQUIRED_RECIPE_DATA_COLUMNS = [
    "id",
    "name",
    "minutes",
    "ingredients",
    *NUTRITION_FIELDS,
]
OPTIONAL_RECIPE_DATA_COLUMNS = ["description", "steps", "tags"]

app = FastAPI(title="Recipe Recommender API")
logger = logging.getLogger(__name__)


class RecommendationRequest(BaseModel):
    ingredients: list[str] = Field(min_length=1, max_length=100)
    max_missing_ingredients: int = Field(default=3, ge=0)
    limit: int = Field(default=10, ge=1, le=100)
    flexible_matching: bool = True
    diet: Literal["vegetarian", "non_vegetarian", "vegan"] = "non_vegetarian"
    exclude_eggs: bool = False
    pantry_ingredients: list[str] | None = None
    cuisine: Literal["all", "indian"] = "all"

    @field_validator("ingredients")
    @classmethod
    def ingredients_must_not_be_blank(cls, ingredients: list[str]) -> list[str]:
        if not any(ingredient.strip() for ingredient in ingredients):
            raise ValueError("ingredients must include at least one non-blank value")
        return ingredients


@lru_cache(maxsize=1)
def _load_recipes() -> pd.DataFrame:
    started = time.perf_counter()
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Processed recipe data not found: {DATASET_PATH}")

    available_columns = pd.read_csv(DATASET_PATH, nrows=0).columns
    missing_columns = [
        column for column in REQUIRED_RECIPE_DATA_COLUMNS if column not in available_columns
    ]
    if missing_columns:
        raise ValueError(f"Processed recipe data is missing columns: {missing_columns}")

    columns_to_load = [
        column for column in RECIPE_DATA_COLUMNS if column in available_columns
    ]
    recipes = pd.read_csv(
        DATASET_PATH,
        usecols=columns_to_load,
        low_memory=False,
    )
    recipes["ingredients"] = recipes["ingredients"].map(parse_list_value)
    for column in OPTIONAL_RECIPE_DATA_COLUMNS:
        if column not in recipes:
            recipes[column] = None if column == "description" else [[] for _ in range(len(recipes))]
    logger.info(
        "Processed recipe dataset loaded: rows=%d elapsed_seconds=%.3f",
        len(recipes),
        time.perf_counter() - started,
    )
    return recipes


@lru_cache(maxsize=1)
def _load_recommender() -> RecipeRecommender:
    started = time.perf_counter()
    logger.info("Initializing cached recipe recommender.")
    recommender = RecipeRecommender(_load_recipes())
    logger.info(
        "Cached recipe recommender ready: elapsed_seconds=%.3f",
        time.perf_counter() - started,
    )
    return recommender


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: RecommendationRequest) -> JSONResponse:
    request_started = time.perf_counter()
    cache_before = _load_recommender.cache_info()
    try:
        recommender = _load_recommender()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Processed recipe data is unavailable") from exc
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=503, detail="Processed recipe data could not be read") from exc

    recommendation_started = time.perf_counter()
    results = recommender.recommend(
        request.ingredients,
        max_missing=request.max_missing_ingredients,
        limit=request.limit,
        flexible_matching=request.flexible_matching,
        diet=request.diet,
        exclude_eggs=request.exclude_eggs,
        pantry_ingredients=request.pantry_ingredients,
        cuisine=request.cuisine,
    )
    logger.info(
        "Recommendation computation completed: result_count=%d cuisine=%s elapsed_seconds=%.3f",
        len(results),
        request.cuisine,
        time.perf_counter() - recommendation_started,
    )
    cache_after = _load_recommender.cache_info()
    serialization_started = time.perf_counter()
    recommendations = []
    for _, recipe in results.iterrows():
        nutrition = {
            field: _json_value(recipe.get(field))
            for field in NUTRITION_FIELDS
        }
        recommendations.append(
            {
                "recipe_id": _json_value(recipe.get("id")),
                "name": _json_value(recipe.get("name")),
                "description": _json_value(recipe.get("description")),
                "cooking_time_minutes": _json_value(recipe.get("minutes")),
                "ingredients": parse_list_value(recipe.get("ingredients")),
                "steps": parse_list_value(recipe.get("steps")),
                "matched_ingredients": recipe["matched_ingredients"],
                "missing_ingredients": recipe["missing_ingredients"],
                "extra_ingredients": recipe["extra_ingredients"],
                "pantry_extras": recipe["pantry_extras"],
                "additional_ingredients": recipe["additional_ingredients"],
                "ingredient_matches": int(recipe["ingredient_matches"]),
                "ingredient_match_ratio": float(recipe["ingredient_match_ratio"]),
                "similarity_score": float(recipe["similarity_score"]),
                "recommendation_score": float(recipe["recommendation_score"]),
                "match_status": recipe["match_status"],
                "dietary_classification": recipe["dietary_classification"],
                "dietary_review_required": bool(recipe["dietary_review_required"]),
                "cuisine_category": recipe["cuisine_category"],
                "cuisine_confidence": float(recipe["cuisine_confidence"]),
                "cuisine_classification_basis": recipe["cuisine_classification_basis"],
                "nutrition": nutrition,
            }
        )

    response_body = {
        "recommendations": recommendations,
        "message": (
            None
            if recommendations
            else (
                "No relevant recipes matched any of your entered ingredients. "
                "Try adding more ingredients or changing your ingredient list."
            )
        ),
        "nutrition_basis": (
            "The dataset does not specify whether nutrition values are per serving. "
            "Fields ending in _pdv are percentages of Daily Value."
        ),
    }
    response = JSONResponse(content=response_body)
    logger.info(
        "Recommendation response serialization completed: result_count=%d bytes=%d elapsed_seconds=%.3f",
        len(recommendations),
        len(response.body),
        time.perf_counter() - serialization_started,
    )
    logger.info(
        "Full POST /recommend completed: result_count=%d recommender_cache_hit=%s total_elapsed_seconds=%.3f",
        len(recommendations),
        cache_after.hits > cache_before.hits,
        time.perf_counter() - request_started,
    )
    return response
