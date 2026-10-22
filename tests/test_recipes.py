import pandas as pd
import pytest

from src.recipes import (
    CLEAN_RECIPE_COLUMNS,
    NUTRITION_FIELDS,
    RECIPE_COLUMNS,
    RecipeRecommender,
    classify_recipe_cuisine,
    clean_recipe_frame,
    parse_ingredient_input,
    parse_list_value,
    parse_nutrition_value,
    recommend_recipes,
)


def test_parse_list_value_handles_valid_malformed_and_missing_values():
    assert parse_list_value("['tomato', ' basil ']") == ["tomato", "basil"]
    assert parse_list_value('["tomato", 3, null]') == []
    assert parse_list_value(None) == []
    assert parse_list_value(float("nan")) == []
    assert parse_list_value([" tomato ", None, ""]) == ["tomato"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "rice.onion.cabbage.tomato",
            ["rice", "onion", "cabbage", "tomato"],
        ),
        (
            "rice, onion, cabbage, tomato",
            ["rice", "onion", "cabbage", "tomato"],
        ),
        (
            "rice\nonion\ncabbage\ntomato",
            ["rice", "onion", "cabbage", "tomato"],
        ),
        (
            "rice; onion; cabbage; tomato",
            ["rice", "onion", "cabbage", "tomato"],
        ),
    ],
)
def test_parse_ingredient_input_supported_separators(value, expected):
    assert parse_ingredient_input(value) == expected


def test_parse_ingredient_input_trims_normalizes_and_deduplicates():
    assert parse_ingredient_input(" Rice, onion; RICE.\n Tomato ") == [
        "rice",
        "onion",
        "tomato",
    ]


def test_clean_recipe_frame_parses_lists_and_preserves_required_columns():
    source = pd.DataFrame(
        [
            {
                "name": "Tomato soup",
                "id": 1,
                "minutes": 20,
                "ingredients": "['tomato', 'water']",
                "steps": "['boil']",
                "description": None,
                "nutrition": "[1, 2]",
                "n_ingredients": 2,
                "ignored": "not included",
            },
            {
                "name": "Broken recipe",
                "id": 2,
                "minutes": 5,
                "ingredients": "not a list",
                "steps": None,
                "description": "",
                "nutrition": "[0]",
                "n_ingredients": 0,
            },
        ]
    )

    cleaned = clean_recipe_frame(source)

    assert cleaned.columns.tolist() == CLEAN_RECIPE_COLUMNS
    assert cleaned.loc[0, "ingredients"] == ["tomato", "water"]
    assert cleaned.loc[0, "steps"] == ["boil"]
    assert cleaned.loc[0, "tags"] == []
    assert pd.isna(cleaned.loc[0, "calories"])
    assert pd.isna(cleaned.loc[0, "cholesterol_pdv"])
    assert pd.isna(cleaned.loc[0, "fiber_pdv"])
    assert cleaned.loc[1, "ingredients"] == []
    assert cleaned.loc[1, "steps"] == []
    assert cleaned.loc[1, "tags"] == []
    assert pd.isna(cleaned.loc[0, "description"])
    assert "ingredients" in source.columns and isinstance(source.loc[0, "ingredients"], str)


def test_clean_recipe_frame_reports_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_recipe_frame(pd.DataFrame({"name": ["incomplete"]}))


def test_parse_nutrition_value_maps_documented_source_order():
    parsed = parse_nutrition_value("[51.5, 10, 20, 30, 40, 50, 60]")

    assert parsed == {
        "calories": 51.5,
        "total_fat_pdv": 10.0,
        "sugar_pdv": 20.0,
        "sodium_pdv": 30.0,
        "protein_pdv": 40.0,
        "saturated_fat_pdv": 50.0,
        "carbohydrates_pdv": 60.0,
        "cholesterol_pdv": None,
        "fiber_pdv": None,
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "not a list",
        "[]",
        "[1, 2, 3]",
        "[1, 2, 3, 4, 5, 6, 7, 8]",
        "[1, 2, 3, 4, 5, 6, -7]",
        "[1, 2, 3, 4, 5, 6, inf]",
        [1, 2, 3, 4, 5, 6, True],
    ],
)
def test_parse_nutrition_value_returns_empty_fields_for_invalid_values(value):
    assert parse_nutrition_value(value) == dict.fromkeys(NUTRITION_FIELDS)


def test_cleaned_nutrition_fields_do_not_break_ingredient_recommendations():
    source = pd.DataFrame(
        [
            {
                "name": "Tomato basil soup",
                "id": 1,
                "minutes": 20,
                "ingredients": "['tomato', 'basil']",
                "steps": "['cook']",
                "description": "",
                "nutrition": "[100, 10, 20, 30, 40, 50, 60]",
                "n_ingredients": 2,
            },
            {
                "name": "Tomato soup",
                "id": 2,
                "minutes": 15,
                "ingredients": "['tomato']",
                "steps": "['cook']",
                "description": "",
                "nutrition": "invalid",
                "n_ingredients": 1,
            },
        ]
    )

    cleaned = clean_recipe_frame(source)
    result = recommend_recipes(
        cleaned,
        ["tomato", "basil"],
        max_missing=1,
        flexible_matching=True,
    )

    assert result["id"].tolist() == [1, 2]
    assert result["ingredient_matches"].tolist() == [2, 1]
    assert cleaned.loc[0, "calories"] == 100.0
    assert pd.isna(cleaned.loc[1, "calories"])


def test_recommend_recipes_prioritizes_matches_and_lists_missing_ingredients():
    recipes = pd.DataFrame(
        [
            {"id": 1, "name": "Two matches", "ingredients": ["tomato", "basil"]},
            {"id": 2, "name": "One match", "ingredients": ["tomato", "cheese"]},
            {"id": 3, "name": "All ingredients", "ingredients": ["tomato", "basil", "onion"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        [" TOMATO ", "basil", "onion"],
        max_missing=1,
        flexible_matching=True,
    )

    assert result["id"].tolist() == [3, 1]
    assert result["ingredient_matches"].tolist() == [3, 2]
    assert result["missing_ingredients"].tolist() == [[], ["onion"]]


def test_recommend_recipes_respects_missing_limit_and_rejects_invalid_limit():
    recipes = pd.DataFrame(
        [{"id": 1, "name": "Tomato", "ingredients": ["tomato"]}]
    )

    assert recommend_recipes(
        recipes,
        ["tomato", "basil"],
        max_missing=0,
        flexible_matching=False,
    ).empty
    assert recommend_recipes(
        recipes,
        ["tomato", "basil"],
        max_missing=1,
        flexible_matching=True,
    ).iloc[0]["missing_ingredients"] == ["basil"]
    with pytest.raises(ValueError, match="non-negative integer"):
        recommend_recipes(recipes, ["tomato"], max_missing=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        recommend_recipes(recipes, ["tomato"], max_missing=True)


def test_recommend_recipes_returns_no_results_without_user_ingredients():
    recipes = pd.DataFrame(
        [{"id": 1, "name": "Tomato", "ingredients": ["tomato"]}]
    )

    assert recommend_recipes(recipes, []).empty


def test_recommend_recipes_normalizes_and_deduplicates_user_ingredients():
    recipes = pd.DataFrame(
        [
            {"id": 1, "ingredients": ["Tomato", "fresh   BASIL"]},
            {"id": 2, "ingredients": ["tomato"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        [" TOMATO ", "tomato", "  Fresh basil  "],
        max_missing=1,
        flexible_matching=True,
    )

    assert result["id"].tolist() == [1, 2]
    assert result["ingredient_matches"].tolist() == [2, 1]
    assert result["matched_ingredients"].tolist() == [
        ["tomato", "fresh basil"],
        ["tomato"],
    ]
    assert result["missing_ingredients"].tolist() == [
        [],
        ["fresh basil"],
    ]


def test_recommend_recipes_missing_lists_and_invalid_rows_are_safe():
    recipes = pd.DataFrame(
        [
            {"recipe": "valid", "ingredients": [" TOMATO "]},
            {"recipe": "missing", "ingredients": None},
            {"recipe": "malformed", "ingredients": "not a list"},
            {"recipe": "wrong type", "ingredients": 42},
            {"recipe": "non-string values", "ingredients": ["tomato", None, 42]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato", "basil"],
        max_missing=2,
        flexible_matching=True,
    )

    assert result["recipe"].tolist() == ["valid", "non-string values"]
    assert result["matched_ingredients"].tolist() == [["tomato"], ["tomato"]]
    assert result["missing_ingredients"].tolist() == [["basil"], ["basil"]]


def test_recommend_recipes_handles_none_and_whitespace_only_input():
    recipes = pd.DataFrame([{"ingredients": ["tomato"]}])

    assert recommend_recipes(recipes, None).empty
    assert recommend_recipes(recipes, [" ", "\t"]).empty


def test_recommend_recipes_keeps_stable_order_when_match_and_missing_counts_tie():
    recipes = pd.DataFrame(
        [
            {"recipe": "first", "ingredients": ["tomato"]},
            {"recipe": "second", "ingredients": ["tomato", "pepper"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato", "basil"],
        max_missing=1,
        flexible_matching=True,
    )

    assert result["recipe"].tolist() == ["first", "second"]
    assert result["ingredient_matches"].tolist() == [1, 1]
    assert result["missing_ingredients"].tolist() == [["basil"], ["basil"]]


def test_recommend_recipes_defaults_to_three_missing_user_ingredients():
    recipes = pd.DataFrame(
        [
            {"recipe": "three missing", "ingredients": ["tomato", "basil"]},
            {"recipe": "four missing", "ingredients": ["tomato"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato", "basil", "onion", "pepper", "garlic"],
        flexible_matching=True,
    )

    assert result["recipe"].tolist() == ["three missing"]
    assert result.iloc[0]["missing_ingredients"] == ["onion", "pepper", "garlic"]


def test_recommend_recipes_prefers_full_match_and_returns_closest_partial_matches():
    recipes = pd.DataFrame(
        [
            {"recipe": "partial", "ingredients": ["tomato", "basil", "water"]},
            {"recipe": "full", "ingredients": ["tomato", "basil", "onion", "salt"]},
            {"recipe": "less close", "ingredients": ["tomato", "water"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        [" TOMATO ", "basil", "onion"],
        max_missing=2,
        flexible_matching=True,
    )

    assert result["recipe"].tolist() == ["full", "partial", "less close"]
    assert result["ingredient_matches"].tolist() == [3, 2, 1]
    assert result["matched_ingredients"].tolist() == [
        ["tomato", "basil", "onion"],
        ["tomato", "basil"],
        ["tomato"],
    ]
    assert result["missing_ingredients"].tolist() == [
        [],
        ["onion"],
        ["basil", "onion"],
    ]


def test_recommend_recipes_ranks_extra_count_before_common_extra_preference():
    recipes = pd.DataFrame(
        [
            {"recipe": "two common extras", "ingredients": ["tomato", "salt", "pepper"]},
            {"recipe": "one rare extra", "ingredients": ["tomato", "saffron"]},
            {"recipe": "one common extra", "ingredients": ["tomato", "garlic"]},
        ]
    )

    result = recommend_recipes(recipes, ["tomato"])

    assert result["recipe"].tolist() == [
        "one common extra",
        "one rare extra",
        "two common extras",
    ]
    assert result["extra_ingredients"].tolist() == [
        ["garlic"],
        ["saffron"],
        ["pepper", "salt"],
    ]


def test_recommend_recipes_filters_by_missing_user_ingredient_limit():
    recipes = pd.DataFrame(
        [
            {"recipe": "one missing", "ingredients": ["tomato", "basil"]},
            {"recipe": "two missing", "ingredients": ["tomato"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato", "basil", "onion"],
        max_missing=1,
        flexible_matching=True,
    )

    assert result["recipe"].tolist() == ["one missing"]
    assert result.iloc[0]["missing_ingredients"] == ["onion"]


def test_tfidf_cosine_similarity_ranks_more_similar_recipes_first():
    recipes = pd.DataFrame(
        [
            {
                "recipe": "similar",
                "ingredients": ["tomato", "basil", "onion", "water"],
            },
            {
                "recipe": "less similar",
                "ingredients": ["tomato", "basil", "onion", "water", "oregano"],
            },
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato", "basil", "onion"],
        max_missing=0,
    )

    assert result["recipe"].tolist() == ["similar", "less similar"]
    assert result["similarity_score"].between(0, 1).all()
    assert result.iloc[0]["similarity_score"] > result.iloc[1]["similarity_score"]


def test_tfidf_index_prioritizes_exact_ingredient_match_over_partial_match():
    recipes = pd.DataFrame(
        [
            {"recipe": "partial", "ingredients": ["tomato", "basil"]},
            {"recipe": "full", "ingredients": ["tomato", "basil", "onion"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato", "basil", "onion"],
        max_missing=1,
        flexible_matching=True,
    )

    assert result["recipe"].tolist() == ["full", "partial"]
    assert result.iloc[0]["matched_ingredients"] == ["tomato", "basil", "onion"]
    assert result.iloc[1]["missing_ingredients"] == ["onion"]


def test_tfidf_index_handles_unknown_ingredients_and_empty_recipe_lists():
    recipes = pd.DataFrame(
        [
            {"recipe": "non-empty", "ingredients": ["tomato"]},
            {"recipe": "empty", "ingredients": None},
            {"recipe": "malformed", "ingredients": "not a list"},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["ingredient-not-in-vocabulary"],
        max_missing=1,
        flexible_matching=True,
    )

    assert result.empty


def test_tfidf_index_handles_dataset_with_only_empty_ingredient_lists():
    recipes = pd.DataFrame(
        [
            {"recipe": "empty one", "ingredients": None},
            {"recipe": "empty two", "ingredients": "[]"},
        ]
    )

    result = RecipeRecommender(recipes).recommend(
        ["tomato"],
        max_missing=1,
        flexible_matching=True,
    )

    assert result.empty


def test_cuisine_classifier_uses_dataset_tags_and_marks_classification_inferred():
    category, confidence, basis = classify_recipe_cuisine(
        "Vegetable stew",
        ["potato", "onion"],
        ["main-dish", "indian"],
    )

    assert category == "Indian"
    assert confidence >= 0.9
    assert basis == "inferred_from_dataset_tags"


def test_cuisine_classifier_uses_distinctive_dish_names_and_ingredient_cues():
    named = classify_recipe_cuisine("Vegetable biryani", ["rice", "onion"], [])
    ingredient_based = classify_recipe_cuisine(
        "Spiced vegetable dish",
        ["garam masala", "curry leaves", "potato"],
        [],
    )

    assert named[0] == ingredient_based[0] == "Indian"
    assert named[2] == "inferred_from_recipe_name"
    assert ingredient_based[2] == "inferred_from_ingredient_cues"


@pytest.mark.parametrize(
    ("name", "ingredients", "tags"),
    [
        ("Potato and tomato soup", ["potato", "tomato", "onion"], []),
        ("Unknown recipe", ["salt", "pepper"], []),
        ("Curry night", ["chicken", "onion"], ["dinner"]),
    ],
)
def test_cuisine_classifier_does_not_overclaim_from_generic_or_ambiguous_data(
    name,
    ingredients,
    tags,
):
    assert classify_recipe_cuisine(name, ingredients, tags) == (
        None,
        0.0,
        "insufficient_evidence",
    )


def test_recommender_can_filter_to_indian_cuisine_without_changing_all_mode():
    recipes = pd.DataFrame(
        [
            {
                "recipe": "Tagged Indian tomato dish",
                "name": "Vegetable dish",
                "ingredients": ["tomato", "onion"],
                "tags": ["indian"],
            },
            {
                "recipe": "Unknown tomato dish",
                "name": "Tomato soup",
                "ingredients": ["tomato", "water"],
                "tags": [],
            },
        ]
    )
    recommender = RecipeRecommender(recipes)

    all_results = recommender.recommend(["tomato"])
    indian_results = recommender.recommend(["tomato"], cuisine="indian")

    assert set(all_results["recipe"]) == {
        "Tagged Indian tomato dish",
        "Unknown tomato dish",
    }
    assert indian_results["recipe"].tolist() == ["Tagged Indian tomato dish"]
    assert indian_results.iloc[0]["cuisine_classification_basis"] == "inferred_from_dataset_tags"
    with pytest.raises(ValueError, match="cuisine must be all or indian"):
        recommender.recommend(["tomato"], cuisine="unknown")


def test_indian_cuisine_mask_is_built_once_and_reused():
    recommender = RecipeRecommender(
        pd.DataFrame(
            [
                {
                    "name": "Vegetable recipe",
                    "ingredients": ["potato", "onion"],
                    "tags": ["indian"],
                },
                {
                    "name": "Besan curry leaves dish",
                    "ingredients": ["besan", "curry leaves"],
                    "tags": [],
                },
                {
                    "name": "Potato tomato soup",
                    "ingredients": ["potato", "tomato"],
                    "tags": [],
                },
            ]
        )
    )

    first_mask = recommender._indian_cuisine_mask()
    second_mask = recommender._indian_cuisine_mask()

    assert first_mask.tolist() == [True, True, False]
    assert second_mask is first_mask


def test_strict_matching_returns_only_recipes_containing_all_user_ingredients():
    recipes = pd.DataFrame(
        [
            {"recipe": "partial", "ingredients": ["tomato", "salt"]},
            {"recipe": "full", "ingredients": ["tomato", "basil", "salt"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        [" TOMATO ", " basil ", "tomato"],
        flexible_matching=False,
    )

    assert result["recipe"].tolist() == ["full"]
    assert result.iloc[0]["match_status"] == "full_match"
    assert result.iloc[0]["matched_ingredients"] == ["tomato", "basil"]
    assert result.iloc[0]["missing_ingredients"] == []
    assert result.iloc[0]["extra_ingredients"] == ["salt"]


def test_flexible_matching_labels_partial_results_and_shows_missing_user_ingredients():
    recipes = pd.DataFrame(
        [
            {"recipe": "partial", "ingredients": ["tomato", "salt"]},
            {"recipe": "full", "ingredients": ["tomato", "basil", "salt"]},
        ]
    )

    strict = recommend_recipes(
        recipes,
        ["tomato", "basil"],
        flexible_matching=False,
    )
    flexible = recommend_recipes(
        recipes,
        ["tomato", "basil"],
        flexible_matching=True,
        max_missing=1,
    )

    assert strict["recipe"].tolist() == ["full"]
    assert flexible["recipe"].tolist() == ["full", "partial"]
    assert flexible.iloc[0]["match_status"] == "full_match"
    assert flexible.iloc[1]["match_status"] == "partial_match"
    assert flexible.iloc[1]["missing_ingredients"] == ["basil"]
    assert flexible.iloc[1]["pantry_extras"] == ["salt"]
    assert flexible.iloc[1]["additional_ingredients"] == []


def test_flexible_matching_is_default_and_returns_partial_recipes():
    recipes = pd.DataFrame(
        [{"recipe": "partial", "ingredients": ["tomato"]}]
    )

    result = recommend_recipes(recipes, ["tomato", "basil"])

    assert result["recipe"].tolist() == ["partial"]
    assert result.iloc[0]["match_status"] == "partial_match"
    assert result.iloc[0]["missing_ingredients"] == ["basil"]


def test_pantry_ingredients_are_reported_separately_and_can_be_configured():
    recipes = pd.DataFrame(
        [
            {"recipe": "pantry extra", "ingredients": ["tomato", "salt"]},
            {"recipe": "major extra", "ingredients": ["tomato", "eggs"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato"],
        pantry_ingredients=["salt"],
    )

    assert result.iloc[0]["recipe"] == "pantry extra"
    assert result.iloc[0]["pantry_extras"] == ["salt"]
    assert result.iloc[0]["additional_ingredients"] == []
    assert result.iloc[1]["pantry_extras"] == []
    assert result.iloc[1]["additional_ingredients"] == ["eggs"]


def test_vegetarian_diet_excludes_meat_but_allows_eggs_unless_excluded():
    recipes = pd.DataFrame(
        [
            {"recipe": "plant", "ingredients": ["tomato", "potatoes", "salt"]},
            {"recipe": "eggs", "ingredients": ["tomato", "eggs"]},
            {"recipe": "meat", "ingredients": ["tomato", "chicken"]},
            {"recipe": "dairy", "ingredients": ["tomato", "paneer"]},
            {"recipe": "uncertain", "ingredients": ["tomato", "mystery ingredient"]},
        ]
    )

    vegetarian = recommend_recipes(recipes, ["tomato"], diet="vegetarian")
    egg_free = recommend_recipes(
        recipes,
        ["tomato"],
        diet="vegetarian",
        exclude_eggs=True,
    )

    assert set(vegetarian["recipe"]) == {"plant", "eggs", "dairy", "uncertain"}
    assert set(egg_free["recipe"]) == {"plant", "dairy", "uncertain"}
    assert bool(vegetarian.loc[vegetarian["recipe"] == "uncertain", "dietary_review_required"].iloc[0])


def test_vegan_filter_excludes_known_animal_and_uncertain_ingredients():
    recipes = pd.DataFrame(
        [
            {"recipe": "plant", "ingredients": ["tomato", "olive oil", "salt"]},
            {"recipe": "eggs", "ingredients": ["tomato", "eggs"]},
            {"recipe": "dairy", "ingredients": ["tomato", "paneer"]},
            {"recipe": "honey", "ingredients": ["tomato", "honey"]},
            {"recipe": "meat", "ingredients": ["tomato", "chicken"]},
            {"recipe": "uncertain", "ingredients": ["tomato", "mystery ingredient"]},
        ]
    )

    result = recommend_recipes(recipes, ["tomato"], diet="vegan")

    assert result["recipe"].tolist() == ["plant"]
    assert result.iloc[0]["dietary_classification"] == "vegan"
    assert not result.iloc[0]["dietary_review_required"]


def test_empty_ingredient_list_is_uncertain_and_not_vegan():
    recipes = pd.DataFrame(
        [
            {"recipe": "empty", "ingredients": None},
            {"recipe": "tomato", "ingredients": ["tomato"]},
        ]
    )

    flexible_results = recommend_recipes(
        recipes,
        ["tomato", "basil"],
        max_missing=2,
        flexible_matching=True,
    )
    vegan_results = recommend_recipes(
        recipes,
        ["tomato", "basil"],
        max_missing=2,
        flexible_matching=True,
        diet="vegan",
    )

    assert "empty" not in flexible_results["recipe"].tolist()
    assert "empty" not in vegan_results["recipe"].tolist()


def test_non_vegetarian_filter_allows_meat_and_flags_uncertain_items():
    recipes = pd.DataFrame(
        [
            {"recipe": "chicken", "ingredients": ["chicken"]},
            {"recipe": "ambiguous", "ingredients": ["mystery ingredient"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["chicken"],
        flexible_matching=True,
        max_missing=1,
        diet="non_vegetarian",
    )

    assert result["recipe"].tolist() == ["chicken"]
    assert result.iloc[0]["dietary_classification"] == "non_vegetarian"


def test_flexible_matching_falls_back_when_no_recipe_meets_missing_limit():
    recipes = pd.DataFrame(
        [
            {"recipe": "closest", "ingredients": ["tomato"]},
            {"recipe": "less close", "ingredients": ["basil"]},
        ]
    )

    result = recommend_recipes(
        recipes,
        ["tomato", "basil", "onion", "garlic", "pepper"],
        max_missing=0,
        limit=1,
    )

    assert result.empty


def test_irrelevant_recipes_with_zero_matches_are_not_returned():
    recipes = pd.DataFrame(
        [
            {
                "recipe": "magic bullet powdered sugar",
                "ingredients": ["powdered sugar", "water"],
            },
            {
                "recipe": "apple cider reduction",
                "ingredients": ["apple cider", "brown sugar"],
            },
            {
                "recipe": "rice cabbage tomato",
                "ingredients": ["rice", "cabbage", "tomato", "onion"],
            },
        ]
    )

    result = recommend_recipes(
        recipes,
        ["rice", "onion", "cabbage", "tomato"],
        limit=10,
    )

    assert result["recipe"].tolist() == ["rice cabbage tomato"]
    assert result.iloc[0]["ingredient_matches"] == 4
    assert result.iloc[0]["similarity_score"] > 0
