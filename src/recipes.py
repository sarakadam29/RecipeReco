import ast
from functools import lru_cache
import logging
import math
import re
import time
from collections.abc import Iterable
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RECIPE_COLUMNS = [
    "name",
    "id",
    "minutes",
    "ingredients",
    "steps",
    "description",
    "nutrition",
    "n_ingredients",
]
OPTIONAL_RECIPE_COLUMNS = ["tags"]

NUTRITION_FIELDS = [
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
    "cholesterol_pdv",
    "fiber_pdv",
]

PARSED_NUTRITION_FIELDS = NUTRITION_FIELDS[:7]
CLEAN_RECIPE_COLUMNS = [*RECIPE_COLUMNS, *OPTIONAL_RECIPE_COLUMNS, *NUTRITION_FIELDS]
logger = logging.getLogger(__name__)
# Defaults are conveniences, not assumptions about a user's pantry. Callers can
# replace this set per recommendation request; substantial foods are excluded.
COMMON_EXTRA_INGREDIENTS = {
    "black pepper",
    "cooking oil",
    "cayenne pepper",
    "chili powder",
    "cinnamon",
    "cloves",
    "cumin",
    "garlic",
    "garlic powder",
    "ginger",
    "kosher salt",
    "coriander",
    "coriander powder",
    "curry powder",
    "dill",
    "fennel",
    "nutmeg",
    "oil",
    "olive oil",
    "oregano",
    "paprika",
    "parsley",
    "pepper",
    "salt",
    "sea salt",
    "soy sauce",
    "sugar",
    "thyme",
    "turmeric",
    "vegetable oil",
    "water",
}

MEAT_OR_FISH_TERMS = {
    "bacon", "beef", "bison", "chicken", "duck", "fish", "goat", "ham",
    "lamb", "meat", "mutton", "pork", "prosciutto", "quail", "rabbit",
    "sausage", "seafood", "shrimp", "turkey", "veal", "venison",
    "anchovy", "anchovies", "clam", "crab", "lobster", "oyster", "salmon",
    "sardine", "scallop", "tuna", "gelatin", "lard",
}
EGG_TERMS = {"egg", "eggs", "albumen", "egg white", "egg yolk", "mayonnaise"}
DAIRY_TERMS = {
    "butter", "buttermilk", "casein", "cheese", "cream", "curd", "ghee",
    "milk", "paneer", "whey", "yogurt", "yoghurt",
}
OTHER_ANIMAL_TERMS = {"honey", "beeswax"}
PLANT_MILK_TERMS = {"almond milk", "coconut milk", "oat milk", "soy milk"}
PLANT_BUTTER_TERMS = {
    "almond butter", "cocoa butter", "coconut butter", "peanut butter",
    "sunflower butter",
}
AMBIGUOUS_TERMS = {
    "broth", "bouillon", "stock", "gelatin", "gelatine", "rennet",
    "natural flavor", "natural flavors", "flavoring", "worcestershire sauce",
    "shortening",
}
KNOWN_PLANT_INGREDIENTS = {
    "almond", "almond butter", "almond milk", "almonds", "apple", "apples",
    "applesauce", "apricot",
    "avocado", "banana", "barley", "basil", "beans", "beet", "bell pepper",
    "black beans", "black pepper", "blueberries", "bread", "broccoli",
    "brown rice", "brown sugar", "cabbage", "carrot", "carrots", "cashews",
    "celery", "chickpeas", "chives", "chocolate", "cilantro", "cinnamon",
    "cocoa butter", "coconut", "coconut butter", "coconut milk", "cocoa",
    "corn", "cornstarch", "couscous",
    "cucumber", "cumin", "curry powder", "dates", "dill", "eggplant",
    "flour", "garlic", "ginger", "grapes", "green beans", "green onions",
    "lentils", "lettuce", "lime", "lemon", "maple syrup",
    "mint", "mushrooms", "mustard", "oats", "olive oil", "onion", "onions",
    "oat milk", "orange", "oregano", "paprika", "parsley", "peanut",
    "peanut butter", "peanuts", "peas",
    "pecans", "pepper", "potato", "potatoes", "quinoa", "raisins", "rice",
    "rosemary", "sage", "salt", "sesame", "shallot", "soy milk", "soy sauce",
    "spinach", "sunflower butter",
    "squash", "strawberries", "sugar", "sweet potato", "thyme", "tomato",
    "tomatoes", "tortilla", "turmeric", "vanilla", "vegetable oil", "vinegar",
    "walnuts", "water", "yeast", "zucchini",
}

_DIET_TERM_SETS = (
    MEAT_OR_FISH_TERMS,
    EGG_TERMS,
    DAIRY_TERMS,
    OTHER_ANIMAL_TERMS,
    PLANT_MILK_TERMS,
    PLANT_BUTTER_TERMS,
    AMBIGUOUS_TERMS,
    KNOWN_PLANT_INGREDIENTS,
)
_DIET_TERM_PATTERNS = {
    id(terms): re.compile(
        rf"(?<![a-z])(?:{'|'.join(re.escape(term) for term in sorted(terms, key=len, reverse=True))})(?![a-z])"
    )
    for terms in _DIET_TERM_SETS
}


def _contains_term(ingredient: str, terms: set[str]) -> bool:
    pattern = _DIET_TERM_PATTERNS.get(id(terms))
    if pattern is None:
        pattern = _pattern_for_terms(tuple(sorted(terms, key=len, reverse=True)))
    return pattern.search(ingredient) is not None


@lru_cache(maxsize=32)
def _pattern_for_terms(terms: tuple[str, ...]) -> re.Pattern:
    return re.compile(
        rf"(?<![a-z])(?:{'|'.join(re.escape(term) for term in terms)})(?![a-z])"
    )


def classify_recipe_diet(ingredients: Iterable[str]) -> tuple[str, bool]:
    """Classify from listed ingredients only; uncertain entries require review.

    Recognized meat/fish, egg, dairy, and other animal-derived ingredients
    determine the class. Unknown or potentially ambiguous ingredient names
    are marked uncertain rather than assumed plant-based. This is not a
    guarantee against cross-contamination or undisclosed components.
    """
    parsed_ingredients = [
        _normalize_ingredient(item)
        for item in parse_list_value(ingredients)
    ]
    if not parsed_ingredients:
        return "uncertain", True

    has_meat_or_fish = any(
        _contains_term(ingredient, MEAT_OR_FISH_TERMS)
        for ingredient in parsed_ingredients
    )
    has_egg = any(_contains_term(ingredient, EGG_TERMS) for ingredient in parsed_ingredients)
    has_dairy = any(
        _contains_term(ingredient, DAIRY_TERMS)
        and not (
            _contains_term(ingredient, PLANT_MILK_TERMS)
            or _contains_term(ingredient, PLANT_BUTTER_TERMS)
        )
        for ingredient in parsed_ingredients
    )
    has_other_animal = any(
        _contains_term(ingredient, OTHER_ANIMAL_TERMS)
        for ingredient in parsed_ingredients
    )

    if has_meat_or_fish:
        return "non_vegetarian", False
    if has_egg or has_dairy or has_other_animal:
        return "vegetarian", False

    unknown = [
        ingredient
        for ingredient in parsed_ingredients
        if not _contains_term(ingredient, KNOWN_PLANT_INGREDIENTS)
        or _contains_term(ingredient, AMBIGUOUS_TERMS)
    ]
    if unknown:
        return "uncertain", True
    return "vegan", False


def parse_list_value(value: object) -> list[str]:
    """Parse a Python-style list from a CSV cell; malformed values become empty lists."""
    if isinstance(value, (list, tuple)):
        parsed = value
    elif not isinstance(value, str) or not value.strip():
        return []
    else:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError):
            return []

    if not isinstance(parsed, (list, tuple)):
        return []
    return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]


def parse_nutrition_value(value: object) -> dict[str, float | None]:
    """Map Food.com nutrition arrays to named fields; unsupported values remain None.

    The source order is calories, total fat (% Daily Value), sugar (% Daily
    Value), sodium (% Daily Value), protein (% Daily Value), saturated fat
    (% Daily Value), and carbohydrates (% Daily Value). The source does not
    establish a serving basis, and it provides no cholesterol or fiber values.
    """
    empty = dict.fromkeys(NUTRITION_FIELDS)
    if isinstance(value, str):
        if not value.strip():
            return empty
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError):
            return empty
    else:
        parsed = value

    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(PARSED_NUTRITION_FIELDS):
        return empty

    numeric_values = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, Real):
            return empty
        number = float(item)
        if not math.isfinite(number) or number < 0:
            return empty
        numeric_values.append(number)

    return {
        **dict(zip(PARSED_NUTRITION_FIELDS, numeric_values)),
        "cholesterol_pdv": None,
        "fiber_pdv": None,
    }


def clean_recipe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in RECIPE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Recipe data is missing required columns: {missing_columns}")

    cleaned = frame.loc[:, RECIPE_COLUMNS].copy()
    for column in OPTIONAL_RECIPE_COLUMNS:
        cleaned[column] = (
            frame[column]
            if column in frame.columns
            else pd.Series([[] for _ in range(len(frame))], index=frame.index)
        )
    for column in ("ingredients", "steps", *OPTIONAL_RECIPE_COLUMNS):
        cleaned[column] = cleaned[column].map(parse_list_value)
    nutrition = pd.DataFrame(
        cleaned["nutrition"].map(parse_nutrition_value).tolist(),
        index=cleaned.index,
    )
    return pd.concat([cleaned, nutrition], axis=1)


def _normalize_ingredient(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def parse_ingredient_input(value: str) -> list[str]:
    """Split common entry formats into normalized, unique ingredient names."""
    if not isinstance(value, str):
        return []
    return list(
        dict.fromkeys(
            normalized
            for item in re.split(r"[,;.\r\n]+", value)
            if (normalized := _normalize_ingredient(item))
        )
    )


_INDIAN_DISH_NAMES = {
    "aloo gobi", "biryani", "chaat", "chapati", "chole", "dal", "daal",
    "dhokla", "dosa", "idli", "kulfi", "lassi", "naan", "pakora", "pakoda",
    "paratha", "pav bhaji", "poha", "rajma", "rasam", "roti", "saag",
    "sambar", "samosa", "tandoori", "tikka masala", "upma", "uttapam",
    "vada", "vindaloo",
}
_INDIAN_INGREDIENT_CUES = {
    "amchur", "asafoetida", "besan", "chat masala", "chaat masala",
    "curry leaves", "garam masala", "hing", "moong dal", "paneer",
    "toor dal", "urad dal",
}
_INDIAN_INGREDIENT_CUE_PATTERN = re.compile(
    rf"(?<![a-z])({'|'.join(re.escape(cue) for cue in sorted(_INDIAN_INGREDIENT_CUES, key=len, reverse=True))})(?![a-z])"
)
_INDIAN_INGREDIENT_CUE_INDICES = {
    cue: index for index, cue in enumerate(sorted(_INDIAN_INGREDIENT_CUES))
}
_INDIAN_DISH_PATTERN = re.compile(
    rf"(?<![a-z])(?:{'|'.join(re.escape(term) for term in sorted(_INDIAN_DISH_NAMES, key=len, reverse=True))})(?![a-z])"
)


def classify_recipe_cuisine(
    name: object,
    ingredients: Iterable[str] | object,
    tags: Iterable[str] | object = None,
) -> tuple[str | None, float, str]:
    """Infer Indian cuisine conservatively from source tags, names, or ingredient cues.

    Food.com tags are user-generated metadata, so even an explicit ``indian`` tag
    is reported as an inference, not a verified origin. Ingredient-only inference
    requires at least two distinct cuisine-specific cues; generic ingredients
    such as onion, potato, and tomato are never sufficient.
    """
    normalized_tags = {
        _normalize_ingredient(tag)
        for tag in parse_list_value(tags)
    }
    title = _normalize_ingredient(name) if isinstance(name, str) else ""
    if "indian" in normalized_tags or "indian cuisine" in normalized_tags:
        return "Indian", 0.95, "inferred_from_dataset_tags"
    if re.search(r"(?<![a-z])indian(?![a-z])", title):
        return "Indian", 0.9, "inferred_from_recipe_name"
    if _INDIAN_DISH_PATTERN.search(title):
        return "Indian", 0.85, "inferred_from_recipe_name"

    ingredient_values = parse_list_value(ingredients)
    ingredient_text = " ".join(
        _normalize_ingredient(item) for item in ingredient_values
    )
    cues = [
        cue for cue in _INDIAN_INGREDIENT_CUES
        if re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", ingredient_text)
    ]
    if len(cues) >= 2:
        return "Indian", 0.75, "inferred_from_ingredient_cues"
    return None, 0.0, "insufficient_evidence"


def _normalize_user_ingredients(user_ingredients: Iterable[str]) -> list[str]:
    if user_ingredients is None:
        return []
    if isinstance(user_ingredients, str):
        user_ingredients = [user_ingredients]
    return list(
        dict.fromkeys(
            normalized
            for item in user_ingredients
            if isinstance(item, str)
            if (normalized := _normalize_ingredient(item))
        )
    )


class RecipeRecommender:
    """Reusable TF-IDF recipe index with ingredient-availability-aware ranking."""

    def __init__(self, recipes: pd.DataFrame):
        started = time.perf_counter()
        self.recipes = recipes.reset_index(drop=True)
        self.original_ingredients = [
            parse_list_value(value)
            for value in self.recipes.get(
                "ingredients", pd.Series(index=self.recipes.index, dtype=object)
            )
        ]
        self.available_ingredients = [
            {
                normalized
                for ingredient in ingredients
                if (normalized := _normalize_ingredient(ingredient))
            }
            for ingredients in self.original_ingredients
        ]
        self.ingredient_to_rows: dict[str, list[int]] = {}
        for row_index, ingredients in enumerate(self.available_ingredients):
            for ingredient in ingredients:
                self.ingredient_to_rows.setdefault(ingredient, []).append(row_index)
        self.ingredient_counts = np.fromiter(
            (len(ingredients) for ingredients in self.available_ingredients),
            dtype=np.int32,
            count=len(self.available_ingredients),
        )
        self._dietary_cache: list[tuple[str, bool, bool] | None] = [
            None
        ] * len(self.recipes)
        self._dietary_masks: dict[tuple[str, bool], np.ndarray] = {}
        self._cuisine_cache: list[tuple[str | None, float, str] | None] = [
            None
        ] * len(self.recipes)
        self._indian_mask: np.ndarray | None = None
        self.vectorizer = TfidfVectorizer(
            analyzer=lambda document: document.split("\n"),
            lowercase=False,
        )
        self.recipe_matrix = None
        if any(self.available_ingredients):
            documents = [
                "\n".join(sorted(ingredients))
                for ingredients in self.available_ingredients
            ]
            self.recipe_matrix = self.vectorizer.fit_transform(documents)
        logger.info(
            "Recommender initialized: recipes=%d vocabulary=%d elapsed_seconds=%.3f",
            len(self.recipes),
            len(getattr(self.vectorizer, "vocabulary_", {})),
            time.perf_counter() - started,
        )

    def _dietary_info(self, row_index: int) -> tuple[str, bool, bool]:
        cached = self._dietary_cache[row_index]
        if cached is None:
            classification, review_required = classify_recipe_diet(
                self.original_ingredients[row_index]
            )
            contains_eggs = any(
                _contains_term(_normalize_ingredient(ingredient), EGG_TERMS)
                for ingredient in self.original_ingredients[row_index]
            )
            cached = classification, review_required, contains_eggs
            self._dietary_cache[row_index] = cached
        return cached

    def _dietary_mask(self, diet: str, exclude_eggs: bool) -> np.ndarray:
        cache_key = (diet, exclude_eggs)
        if cache_key not in self._dietary_masks:
            mask = np.ones(len(self.recipes), dtype=bool)
            if diet != "non_vegetarian":
                for row_index in range(len(self.recipes)):
                    classification, _, contains_eggs = self._dietary_info(row_index)
                    if diet == "vegan":
                        mask[row_index] = classification == "vegan"
                    else:
                        mask[row_index] = classification != "non_vegetarian"
                        if exclude_eggs and contains_eggs:
                            mask[row_index] = False
            self._dietary_masks[cache_key] = mask
            logger.info(
                "Dietary filter index ready: diet=%s exclude_eggs=%s eligible=%d",
                diet,
                exclude_eggs,
                int(mask.sum()),
            )
        return self._dietary_masks[cache_key]

    def _cuisine_info(self, row_index: int) -> tuple[str | None, float, str]:
        cached = self._cuisine_cache[row_index]
        if cached is None:
            row = self.recipes.iloc[row_index]
            cached = classify_recipe_cuisine(
                row.get("name"),
                self.original_ingredients[row_index],
                row.get("tags"),
            )
            self._cuisine_cache[row_index] = cached
        return cached

    def _indian_cuisine_mask(self) -> np.ndarray:
        if self._indian_mask is None:
            started = time.perf_counter()
            count = len(self.recipes)
            tags = self.recipes.get(
                "tags", pd.Series("", index=self.recipes.index, dtype=object)
            ).fillna("").astype(str)
            names = self.recipes.get(
                "name", pd.Series("", index=self.recipes.index, dtype=object)
            ).fillna("").astype(str)
            tag_mask = tags.str.contains(
                r"""(?<![a-z])['"]indian['"](?![a-z])""",
                case=False,
                regex=True,
                na=False,
            ).to_numpy(dtype=bool)
            name_mask = (
                names.str.casefold().str.contains(
                    r"(?<![a-z])indian(?![a-z])", regex=True, na=False
                )
                | names.str.casefold().str.contains(
                    _INDIAN_DISH_PATTERN, na=False
                )
            ).to_numpy(dtype=bool)
            cuisine_mask = tag_mask | name_mask
            cue_presence = np.zeros(
                (count, len(_INDIAN_INGREDIENT_CUE_INDICES)),
                dtype=bool,
            )
            cue_scan_started = time.perf_counter()
            for ingredient, rows in self.ingredient_to_rows.items():
                cue_indices = {
                    _INDIAN_INGREDIENT_CUE_INDICES[match]
                    for match in _INDIAN_INGREDIENT_CUE_PATTERN.findall(ingredient)
                }
                if cue_indices:
                    row_indices = np.asarray(rows, dtype=np.intp)
                    for cue_index in cue_indices:
                        cue_presence[row_indices, cue_index] = True
            ingredient_mask = cue_presence.sum(axis=1) >= 2
            self._indian_mask = cuisine_mask | ingredient_mask
            logger.info(
                "Indian cuisine index ready: eligible=%d tag_or_name=%d ingredient_cue=%d cue_scan_seconds=%.3f elapsed_seconds=%.3f",
                int(self._indian_mask.sum()),
                int(cuisine_mask.sum()),
                int(ingredient_mask.sum()),
                time.perf_counter() - cue_scan_started,
                time.perf_counter() - started,
            )
        return self._indian_mask

    def recommend(
        self,
        user_ingredients: Iterable[str],
        max_missing: int = 3,
        limit: int | None = None,
        flexible_matching: bool = True,
        diet: str = "non_vegetarian",
        exclude_eggs: bool = False,
        pantry_ingredients: Iterable[str] | None = None,
        cuisine: str = "all",
    ) -> pd.DataFrame:
        if isinstance(max_missing, bool) or not isinstance(max_missing, int) or max_missing < 0:
            raise ValueError("max_missing must be a non-negative integer")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be a positive integer")
        if not isinstance(diet, str) or diet not in {
            "vegetarian",
            "non_vegetarian",
            "vegan",
        }:
            raise ValueError("diet must be vegetarian, non_vegetarian, or vegan")
        if not isinstance(cuisine, str) or cuisine not in {"all", "indian"}:
            raise ValueError("cuisine must be all or indian")

        requested = _normalize_user_ingredients(user_ingredients)
        pantry = set(COMMON_EXTRA_INGREDIENTS)
        if pantry_ingredients is not None:
            pantry = set(_normalize_user_ingredients(pantry_ingredients))
        result_columns = [
            *self.recipes.columns,
            "ingredient_matches",
            "matched_ingredients",
            "missing_ingredients",
            "extra_ingredients",
            "pantry_extras",
            "additional_ingredients",
            "ingredient_match_ratio",
            "similarity_score",
            "recommendation_score",
            "match_status",
            "dietary_classification",
            "dietary_review_required",
            "cuisine_category",
            "cuisine_confidence",
            "cuisine_classification_basis",
        ]
        if not requested:
            return pd.DataFrame(columns=result_columns)

        similarity_started = time.perf_counter()
        similarities = np.zeros(len(self.recipes), dtype=float)
        if self.recipe_matrix is not None:
            query_vector = self.vectorizer.transform(["\n".join(requested)])
            if query_vector.nnz:
                # Query-to-recipe only: this is a 1 x N similarity vector, not N x N.
                similarities = cosine_similarity(query_vector, self.recipe_matrix).ravel()
        logger.info(
            "Query-to-recipe cosine similarity completed: recipes=%d elapsed_seconds=%.3f",
            len(self.recipes),
            time.perf_counter() - similarity_started,
        )

        requested_set = set(requested)
        matching_started = time.perf_counter()
        matched_counts = np.zeros(len(self.recipes), dtype=np.int32)
        for ingredient in requested:
            rows = self.ingredient_to_rows.get(ingredient, ())
            if rows:
                matched_counts[np.asarray(rows, dtype=np.intp)] += 1
        missing_counts = len(requested) - matched_counts
        logger.info(
            "Ingredient matching completed: ingredients=%d elapsed_seconds=%.3f",
            len(requested),
            time.perf_counter() - matching_started,
        )

        filtering_started = time.perf_counter()
        dietary_mask = self._dietary_mask(diet, exclude_eggs).copy()
        if cuisine == "indian":
            dietary_mask &= self._indian_cuisine_mask()
        dietary_mask &= self.ingredient_counts > 0
        eligible_mask = dietary_mask & (matched_counts > 0)
        if not flexible_matching:
            eligible_mask = dietary_mask & (missing_counts == 0)
        elif max_missing < len(requested):
            eligible_mask = dietary_mask & (missing_counts <= max_missing)

        candidate_indices = np.flatnonzero(eligible_mask)
        logger.info(
            "Candidate filtering completed: diet=%s candidates=%d elapsed_seconds=%.3f",
            diet,
            len(candidate_indices),
            time.perf_counter() - filtering_started,
        )

        ranking_started = time.perf_counter()
        if len(candidate_indices):
            matched = matched_counts[candidate_indices]
            missing = missing_counts[candidate_indices]
            extra_counts = self.ingredient_counts[candidate_indices] - matched
            similarity_values = np.asarray(similarities)[candidate_indices]
            match_ratios = matched / len(requested)
            recommendation_scores = 0.5 * similarity_values + 0.5 * match_ratios
            pantry_counts = np.zeros(len(self.recipes), dtype=np.int32)
            for ingredient in pantry - requested_set:
                rows = self.ingredient_to_rows.get(ingredient, ())
                if rows:
                    pantry_counts[np.asarray(rows, dtype=np.intp)] += 1
            additional_counts = extra_counts - pantry_counts[candidate_indices]
            ranked_positions = np.lexsort(
                (
                    candidate_indices,
                    additional_counts,
                    extra_counts,
                    missing,
                    -similarity_values,
                    -matched,
                )
            )
            selected_positions = (
                ranked_positions
                if limit is None
                else ranked_positions[:limit]
            )

            ranked_rows = []
            for position in selected_positions:
                row_index = int(candidate_indices[position])
                available = self.available_ingredients[row_index]
                matched_ingredients = [
                    ingredient
                    for ingredient in requested
                    if ingredient in available
                ]
                missing_ingredients = [
                    ingredient
                    for ingredient in requested
                    if ingredient not in available
                ]
                dietary_classification, review_required, _ = self._dietary_info(row_index)
                cuisine_category, cuisine_confidence, cuisine_basis = self._cuisine_info(row_index)
                extras = sorted(available - requested_set)
                pantry_extras = sorted(available & pantry - requested_set)
                additional = sorted(available - pantry - requested_set)
                row = self.recipes.iloc[row_index].to_dict()
                row.update(
                    {
                        "ingredient_matches": int(matched[position]),
                        "matched_ingredients": matched_ingredients,
                        "missing_ingredients": missing_ingredients,
                        "extra_ingredients": extras,
                        "pantry_extras": pantry_extras,
                        "additional_ingredients": additional,
                        "ingredient_match_ratio": float(match_ratios[position]),
                        "similarity_score": float(similarity_values[position]),
                        "recommendation_score": float(recommendation_scores[position]),
                        "match_status": "partial_match" if missing_ingredients else "full_match",
                        "dietary_classification": dietary_classification,
                        "dietary_review_required": bool(review_required),
                        "cuisine_category": cuisine_category,
                        "cuisine_confidence": float(cuisine_confidence),
                        "cuisine_classification_basis": cuisine_basis,
                    }
                )
                ranked_rows.append(row)
        else:
            ranked_rows = []
        logger.info(
            "Candidate ranking completed: results=%d elapsed_seconds=%.3f",
            len(ranked_rows),
            time.perf_counter() - ranking_started,
        )
        return pd.DataFrame(
            ranked_rows,
            columns=result_columns,
        )


def recommend_recipes(
    recipes: pd.DataFrame,
    user_ingredients: Iterable[str],
    max_missing: int = 3,
    limit: int | None = None,
    flexible_matching: bool = True,
    diet: str = "non_vegetarian",
    exclude_eggs: bool = False,
    pantry_ingredients: Iterable[str] | None = None,
    cuisine: str = "all",
) -> pd.DataFrame:
    """Build a TF-IDF index and use flexible partial matching by default."""
    return RecipeRecommender(recipes).recommend(
        user_ingredients,
        max_missing=max_missing,
        limit=limit,
        flexible_matching=flexible_matching,
        diet=diet,
        exclude_eggs=exclude_eggs,
        pantry_ingredients=pantry_ingredients,
        cuisine=cuisine,
    )
