import html
import json
import logging
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from src.recipes import parse_ingredient_input

API_BASE_URL = os.getenv("RECIPERECO_API_URL", "http://127.0.0.1:8000").rstrip("/")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RecipeReco — Cook with what you have",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ivory: #f6f4ed;
        --paper: #fffefa;
        --sage: #617b68;
        --sage-dark: #415b49;
        --sage-pale: #e8efe7;
        --charcoal: #252a26;
        --muted: #727a72;
        --line: #e7e5dc;
        --clay: #f5e9df;
        --clay-ink: #8b5a3c;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--ivory);
        color: var(--charcoal);
        font-family: 'Aptos', 'Segoe UI', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, .serif {
        font-family: 'Georgia', 'Times New Roman', serif !important;
        color: var(--charcoal);
        letter-spacing: -0.025em;
    }
    .brand-row {
        display:flex; align-items:center; justify-content:space-between;
        border-bottom:1px solid var(--line); padding:0.2rem 0 1.1rem;
        margin-bottom:2.2rem;
    }
    .brand { font-weight:700; font-size:1.15rem; letter-spacing:-0.04em; color:var(--sage-dark); }
    .brand-mark {
        display:inline-flex; width:30px; height:30px; border-radius:10px;
        align-items:center; justify-content:center; margin-right:8px;
        background:var(--sage); color:white; font-size:0.95rem;
    }
    .tagline { color:var(--muted); font-size:0.88rem; }
    .hero {
        padding:2.5rem 0 1.5rem; max-width:760px;
    }
    .eyebrow {
        text-transform:uppercase; color:var(--sage); letter-spacing:.16em;
        font-size:.72rem; font-weight:700; margin-bottom:.8rem;
    }
    .hero h1 { font-size:clamp(2.5rem, 6vw, 4.5rem); line-height:1.04; margin:0; }
    .hero p { color:#626b63; font-size:1.05rem; line-height:1.7; max-width:620px; margin-top:1rem; }
    .input-panel {
        background:var(--paper); border:1px solid var(--line); border-radius:22px;
        padding:1.25rem 1.4rem .6rem !important; margin:1.2rem 0 1.5rem;
        box-shadow:0 10px 35px rgba(42,50,42,.035);
    }
    [data-testid="stForm"] {
        background:var(--paper); border:1px solid var(--line); border-radius:22px;
        padding:1.25rem 1.4rem .6rem; margin:1.2rem 0 1.5rem;
        box-shadow:0 10px 35px rgba(42,50,42,.035);
    }
    div[data-testid="stTextArea"] textarea {
        background:#fffefa; border:1px solid var(--line); border-radius:12px;
        color:var(--charcoal); min-height:100px;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color:var(--sage); box-shadow:0 0 0 2px rgba(97,123,104,.12);
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background:var(--sage); border:0; border-radius:999px; color:white;
        font-weight:700; min-height:3rem; padding:0 1.5rem;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background:var(--sage-dark); border:0;
    }
    div[data-testid="stButton"] button {
        border-radius:999px; border-color:var(--line); color:var(--sage-dark);
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background:var(--paper); border-color:var(--line); border-radius:10px;
    }
    [data-testid="stExpander"] {
        border:1px solid var(--line); border-radius:14px; background:rgba(255,254,250,.6);
    }
    .section-heading { margin:2.4rem 0 1rem; }
    .section-heading h2 { margin:0; font-size:1.9rem; }
    .section-heading p { color:var(--muted); margin:.3rem 0 0; }
    .recipe-card {
        background:var(--paper); border:1px solid var(--line); border-radius:20px;
        padding:1.25rem 1.35rem 1.05rem; min-height:290px; margin-bottom:1.1rem;
        box-shadow:0 8px 24px rgba(42,50,42,.035);
    }
    .card-top { display:flex; justify-content:space-between; align-items:center; gap:10px; }
    .food-placeholder {
        height:48px; width:48px; border-radius:15px; display:flex; align-items:center;
        justify-content:center; background:linear-gradient(145deg,#e8eee5,#f4eadf);
        color:var(--sage-dark); font-size:1.25rem; flex:none;
    }
    .status {
        display:inline-block; border-radius:999px; padding:5px 10px; font-size:.7rem;
        font-weight:700; letter-spacing:.03em; text-transform:uppercase;
    }
    .status-full { background:var(--sage-pale); color:var(--sage-dark); }
    .status-partial { background:var(--clay); color:var(--clay-ink); }
    .recipe-title {     font:600 1.35rem/1.25 'Georgia','Times New Roman',serif; margin:1rem 0 .35rem; }
    .meta { color:var(--muted); font-size:.82rem; }
    .score-row { display:flex; justify-content:space-between; align-items:baseline; margin:1rem 0 .3rem; }
    .score-label { color:var(--muted); font-size:.78rem; }
    .score-value { color:var(--sage-dark); font-size:1rem; font-weight:700; }
    .score-track { height:5px; border-radius:9px; background:#e9e8e0; overflow:hidden; }
    .score-fill { height:100%; background:var(--sage); border-radius:9px; }
    .diet-note { color:var(--muted); font-size:.75rem; margin-top:.65rem; }
    .ingredient-label { color:var(--muted); font-size:.73rem; font-weight:700; margin:.8rem 0 .35rem; }
    .pills { display:flex; flex-wrap:wrap; gap:6px; }
    .pill { border-radius:999px; padding:5px 9px; font-size:.72rem; line-height:1.25; }
    .pill-match { background:var(--sage-pale); color:var(--sage-dark); }
    .pill-missing { background:var(--clay); color:var(--clay-ink); }
    .pill-extra { background:#efeee9; color:#545b55; }
    .nutrition-line {
        color:#687068; font-size:.77rem; border-top:1px solid var(--line);
        margin-top:.85rem; padding-top:.7rem; line-height:1.6;
    }
    .empty-state {
        padding:2.2rem; text-align:center; border:1px dashed #d5d8ce;
        background:rgba(255,254,250,.55); border-radius:20px; color:var(--muted);
    }
    .empty-state strong { color:var(--charcoal); display:block; margin-bottom:.4rem; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
        .brand-row { margin-bottom:1rem; }
        .tagline { font-size:.75rem; }
        .hero { padding:1.4rem 0 1rem; }
        .input-panel { padding:1rem .9rem .45rem; border-radius:16px; }
        .recipe-card { min-height:0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _api_get_health() -> tuple[bool, str]:
    try:
        with urlopen(f"{API_BASE_URL}/health", timeout=3) as response:
            if response.status == 200:
                return True, "API connected"
            return False, f"API returned HTTP {response.status}"
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def _request_recommendations(payload: dict) -> dict:
    started = time.perf_counter()
    logger.info(
        "Sending POST /recommend: url=%s ingredient_count=%d cuisine=%s limit=%s",
        f"{API_BASE_URL}/recommend",
        len(payload.get("ingredients", [])),
        payload.get("cuisine", "all"),
        payload.get("limit", 10),
    )
    request = Request(
        f"{API_BASE_URL}/recommend",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        logger.info(
            "POST /recommend completed: result_count=%d elapsed_seconds=%.3f",
            len(body.get("recommendations", [])),
            time.perf_counter() - started,
        )
        return body
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
        logger.exception(
            "POST /recommend failed: url=%s elapsed_seconds=%.3f",
            f"{API_BASE_URL}/recommend",
            time.perf_counter() - started,
        )
        raise


def _escape(value: object) -> str:
    return html.escape(str(value))


def _pills(values: list[str], kind: str, empty_message: str) -> str:
    if not values:
        return f'<span class="meta">{_escape(empty_message)}</span>'
    return '<div class="pills">' + "".join(
        f'<span class="pill pill-{kind}">{_escape(value)}</span>'
        for value in values
    ) + "</div>"


def _nutrition_summary(nutrition: dict) -> str:
    visible = []
    for field, label in (
        ("calories", "Calories"),
        ("total_fat_pdv", "Fat %DV"),
        ("sugar_pdv", "Sugar %DV"),
        ("sodium_pdv", "Sodium %DV"),
        ("protein_pdv", "Protein %DV"),
        ("carbohydrates_pdv", "Carbs %DV"),
    ):
        value = nutrition.get(field)
        if value is not None:
            visible.append(f"{label} {_escape(value)}")
    return " · ".join(visible) if visible else "Nutrition values unavailable"


def _render_recipe_card(recipe: dict, nutrition_basis: str) -> None:
    match_status = recipe.get("match_status", "partial_match")
    status_class = "status-full" if match_status == "full_match" else "status-partial"
    status_label = "Full match" if match_status == "full_match" else "Partial match"
    name = _escape(recipe.get("name") or "Untitled recipe")
    cooking_time = recipe.get("cooking_time_minutes")
    cooking_label = (
        f"{_escape(cooking_time)} min"
        if cooking_time is not None
        else "Cooking time unavailable"
    )
    diet = _escape(recipe.get("dietary_classification", "unclassified").replace("_", " "))
    cuisine_category = recipe.get("cuisine_category")
    cuisine_label = (
        f" &nbsp;·&nbsp; {_escape(cuisine_category)} (inferred)"
        if cuisine_category
        else ""
    )
    similarity = max(0.0, min(1.0, float(recipe.get("similarity_score") or 0)))
    similarity_percent = round(similarity * 100)
    recommendation = max(
        0.0,
        min(1.0, float(recipe.get("recommendation_score") or similarity)),
    )
    recommendation_percent = round(recommendation * 100)
    card = f"""
    <article class="recipe-card">
      <div class="card-top">
        <div class="food-placeholder" aria-hidden="true">✳</div>
        <span class="status {status_class}">{status_label}</span>
      </div>
      <h3 class="recipe-title">{name}</h3>
      <div class="meta">{cooking_label} &nbsp;·&nbsp; {diet.title()}{cuisine_label}</div>
      <div class="score-row">
        <span class="score-label">Ingredient relevance · combined match and similarity</span>
        <span class="score-value">{recommendation_percent}%</span>
      </div>
      <div class="score-track" aria-label="Ingredient relevance {recommendation_percent}%">
        <div class="score-fill" style="width:{recommendation_percent}%"></div>
      </div>
      <div class="diet-note">Cosine similarity: {similarity_percent}% · Not a model accuracy score</div>
      <div class="ingredient-label">Matched ingredients</div>
      {_pills(recipe.get("matched_ingredients", []), "match", "No entered ingredients matched")}
      <div class="ingredient-label">Missing from recipe</div>
      {_pills(recipe.get("missing_ingredients", []), "missing", "None")}
      <div class="ingredient-label">Additional ingredients</div>
      {_pills(recipe.get("extra_ingredients", []), "extra", "None")}
      <div class="nutrition-line">{_nutrition_summary(recipe.get("nutrition", {}))}</div>
    </article>
    """
    st.markdown(card, unsafe_allow_html=True)

    recipe_id = recipe.get("recipe_id", "recipe")
    with st.expander("View details", expanded=False):
        description = recipe.get("description")
        st.markdown("**Description**")
        st.write(description if isinstance(description, str) and description.strip() else "No description is available in the dataset.")

        st.markdown("**Cuisine classification**")
        cuisine = recipe.get("cuisine_category")
        if cuisine:
            evidence_score = round(float(recipe.get("cuisine_confidence") or 0) * 100)
            basis_label = recipe.get("cuisine_classification_basis", "inferred")
            st.write(f"{cuisine} — inferred (evidence indicator {evidence_score}/100; {basis_label.replace('_', ' ')}).")
            st.caption("This heuristic indicator is not a probability. Cuisine is not a verified regional origin or authenticity claim.")
        else:
            st.write("Cuisine is unknown; the available metadata does not support a confident classification.")

        st.markdown("**Full ingredient list**")
        ingredients = recipe.get("ingredients", [])
        if ingredients:
            st.caption("Ingredient text and quantities are shown as provided by the source; none are inferred.")
            st.markdown("\n".join(f"- {html.escape(str(item))}" for item in ingredients))
        else:
            st.write("No ingredient list or quantities are available in the dataset.")

        st.markdown("**Cooking instructions**")
        steps = recipe.get("steps", [])
        if steps:
            st.markdown("\n".join(
                f"{index}. {html.escape(str(step))}"
                for index, step in enumerate(steps, start=1)
            ))
        else:
            st.write("No cooking instructions are available in the dataset.")

        st.markdown("**Ingredient match**")
        st.markdown("Matched: " + (", ".join(recipe.get("matched_ingredients", [])) or "None"))
        st.markdown("Missing from recipe: " + (", ".join(recipe.get("missing_ingredients", [])) or "None"))
        st.markdown("Additional: " + (", ".join(recipe.get("extra_ingredients", [])) or "None"))

        dietary = recipe.get("dietary_classification", "unclassified").replace("_", " ")
        st.markdown(f"**Dietary classification:** {html.escape(dietary.title())}")
        if recipe.get("dietary_review_required"):
            st.warning(
                "Ingredient data is incomplete or ambiguous. Review the full recipe "
                "before relying on its dietary classification."
            )

        st.markdown("**Nutrition data**")
        nutrition = recipe.get("nutrition", {})
        nutrients = [
            (label, nutrition.get(field))
            for field, label in (
                ("calories", "Calories (dataset unit)"),
                ("total_fat_pdv", "Total fat (% Daily Value)"),
                ("saturated_fat_pdv", "Saturated fat (% Daily Value)"),
                ("cholesterol_pdv", "Cholesterol (% Daily Value)"),
                ("sodium_pdv", "Sodium (% Daily Value)"),
                ("carbohydrates_pdv", "Carbohydrates (% Daily Value)"),
                ("fiber_pdv", "Fiber (% Daily Value)"),
                ("sugar_pdv", "Sugar (% Daily Value)"),
                ("protein_pdv", "Protein (% Daily Value)"),
            )
            if nutrition.get(field) is not None
        ]
        if nutrients:
            for label, value in nutrients:
                st.write(f"{label}: {value}")
        else:
            st.caption("No nutrition values are available for this recipe.")
        st.caption(nutrition_basis)
        st.caption(f"Recipe ID: {recipe_id}")


st.markdown(
    """
    <header class="brand-row">
      <div class="brand"><span class="brand-mark">R</span>RecipeReco</div>
      <div class="tagline">Good food starts with what you have.</div>
    </header>
    <section class="hero">
      <div class="eyebrow">A little inspiration for tonight</div>
      <h1>What’s in your kitchen?</h1>
      <p>Tell us what you have on hand. We’ll find recipes that make the most of your ingredients, with practical alternatives when you need them.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

health_ok, health_message = _api_get_health()
if health_ok:
    st.caption(f"● {health_message}")
else:
    st.warning(
        f"Recipe service is not connected at {API_BASE_URL}. Start the FastAPI server "
        "to search recipes."
    )

with st.form("recipe_search"):
    ingredients_text = st.text_area(
        "Your ingredients",
        placeholder="e.g. tomatoes, chickpeas, spinach, garlic",
        help="Separate ingredients with commas or put one on each line.",
        label_visibility="visible",
    )
    st.caption("Try: tomato · rice · garlic · spinach")
    diet_labels = {
        "Vegetarian": "vegetarian",
        "Non-vegetarian": "non_vegetarian",
        "Vegan": "vegan",
    }
    selected_diet = st.selectbox(
        "Dietary preference",
        options=list(diet_labels),
        index=1,
    )
    cuisine_labels = {"All cuisines": "all", "Indian": "indian"}
    selected_cuisine = st.selectbox(
        "Cuisine",
        options=list(cuisine_labels),
        index=0,
        help="Indian cuisine is inferred from dataset tags, names, and distinctive ingredient combinations.",
    )
    exclude_eggs = False
    if selected_diet == "Vegetarian":
        exclude_eggs = st.checkbox("Exclude eggs")

    with st.expander("More filters", expanded=False):
        flexible_matching = st.checkbox(
            "Include partial matches when ingredients are missing",
            value=True,
            help="When off, only recipes containing every entered ingredient are returned.",
        )
        max_missing = st.slider(
            "Maximum missing ingredients",
            min_value=0,
            max_value=10,
            value=3,
            help="Used when partial matching is enabled.",
        )
        result_limit = st.slider("Number of recipes", 1, 30, 10)
    submitted = st.form_submit_button(
        "Find recipes",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.session_state.pop("recipe_results", None)
    st.session_state.pop("nutrition_basis", None)
    st.session_state.pop("recipe_query", None)
    st.session_state.pop("recipe_message", None)
    entered = parse_ingredient_input(ingredients_text)
    if not entered:
        st.info("Add at least one ingredient to get started.")
    else:
        payload = {
            "ingredients": entered,
            "diet": diet_labels[selected_diet],
            "cuisine": cuisine_labels[selected_cuisine],
            "exclude_eggs": exclude_eggs,
            "flexible_matching": flexible_matching,
            "max_missing_ingredients": max_missing,
            "limit": result_limit,
        }
        try:
            with st.spinner("Finding recipes that fit your kitchen..."):
                if not health_ok:
                    raise ConnectionError("The recipe service health check failed.")
                response_data = _request_recommendations(payload)
            recipes = response_data.get("recommendations", [])
            st.session_state["recipe_results"] = recipes
            st.session_state["nutrition_basis"] = response_data.get("nutrition_basis", "")
            st.session_state["recipe_message"] = response_data.get("message")
            st.session_state["recipe_query"] = entered
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            st.error(f"Recipe service returned HTTP {exc.code}: {detail}")
        except (URLError, TimeoutError, OSError, ConnectionError, json.JSONDecodeError) as exc:
            st.error(f"Could not reach the recipe service. Check that FastAPI is running. ({exc})")

if "recipe_results" in st.session_state:
    recipes = st.session_state["recipe_results"]
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="eyebrow">Your kitchen, reimagined</div>
          <h2>Recipes to explore</h2>
          <p>{len(recipes)} suggestions for {html.escape(", ".join(st.session_state.get("recipe_query", [])))}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not recipes:
        no_results_message = st.session_state.get("recipe_message") or (
            "No relevant recipes matched any of your entered ingredients. "
            "Try adding more ingredients or changing your ingredient list."
        )
        st.info(no_results_message)
        st.caption("Try adding more ingredients or changing your ingredient list.")
    else:
        columns = st.columns(2, gap="large")
        basis = st.session_state.get("nutrition_basis", "")
        for index, recipe in enumerate(recipes):
            with columns[index % 2]:
                _render_recipe_card(recipe, basis)
    st.caption(
        "Similarity scores help order these results; they are not "
        "measures of recipe quality or prediction accuracy."
    )
else:
    st.markdown(
        """
        <div class="empty-state">
          <strong>Your next good meal is already in the kitchen.</strong>
          Add a few ingredients above and RecipeReco will find a place to start.
        </div>
        """,
        unsafe_allow_html=True,
    )
