# %%
# --- Imports ---
import pandas as pd
import torch
import requests
from tqdm import tqdm
from io import BytesIO
import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv, find_dotenv
import os
import random
import json
import google.generativeai as genai
from huggingface_hub import hf_hub_download
import uuid
import logging

logging.basicConfig(
    level=logging.DEBUG,  # or INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,  # <<< MAKE SURE this is set (forces logging config to apply)
)

# %%
# --- Constants ---
mood_labels = [
    "adventurous",
    "comforting",
    "energizing",
    "romantic",
    "cozy",
    "festive",
    "indulgent",
    "refreshing",
]
REPO_ID = "tlockhart/philly_reviews_with_mood.parquet"
FILE_NAME = "philly_reviews_with_mood.parquet"
CACHE_PATH = "mood_by_business_cache.parquet"

# %% [markdown]
# ## Configurations - Enter API KEY

# %%
# --- Model Setup ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
genai.configure(api_key="AIzaSyBHn7xZshvAeFe5ZBLV-30OyZIbSwb9BLE")

# %% [markdown]
# ## Gemini Model Set Up

# %% [markdown]
# ### Base Model - Simply Using Gemeni to Recommend Restaurants


# %%
# --- Gemini Schema & Prompt ---
class RestaurantDetail(BaseModel):
    phone: str = Field(description="Phone")
    address: str = Field(description="Address")
    summary: str = Field(description="Summary")
    moods: str = Field(description="Moods")
    highlight: str = Field(description="Highlight")


# --- Gemini Functions ---
def get_llm_generated_recommendation(mood):
    prompt = f"""
    A user is feeling {mood}. Recommend one unique, real restaurant in Philadelphia that matches this mood.
    Avoid repeating suggestions like 'Talula's Garden'.
    Only return the restaurant name. Do not include any extra text or commentary.
    Session ID: {uuid.uuid4()}
    """.strip()

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        name = response.text.strip().split("\n")[0]
        print("✅ Gemini recommended restaurant:", name)
        return name
    except Exception as e:
        print("❌ Gemini recommendation failed:", e)
        return "Vedge"


def fetch_restaurant_details(name):
    prompt = f"""
You are a helpful assistant. Return structured restaurant info about "{name}" in Philadelphia.

Respond ONLY in **valid JSON format** exactly like this:

{{
  "phone": "(215) 555-1234",
  "address": "123 Walnut St, Philadelphia, PA 19106",
  "summary": "A brief description of the restaurant...",
  "moods": "e.g. romantic, cozy, festive",
  "highlight": "e.g. known for its rooftop bar or award-winning wine list"
}}

Do NOT include any bullet points, explanations, commentary, or text outside the JSON.
""".strip()

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        print("🪵 Gemini raw text:\n", raw_text)

        # Strip markdown-style JSON block if present
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        # Try to load JSON strictly
        data = json.loads(raw_text)

        # Basic field validation
        required_fields = ["phone", "address", "summary", "moods", "highlight"]
        if not all(
            field in data and isinstance(data[field], str) for field in required_fields
        ):
            raise ValueError("Missing required fields in Gemini response")

        print("📦 Gemini returned clean structured info.")
        return RestaurantDetail(**data)

    except Exception as e:
        print(f"❌ Failed to get details for {name}: {e}")
        return RestaurantDetail(
            phone="Unknown",
            address="(Generated) Philadelphia",
            summary="Recommended by Gemini for the mood",
            moods="Gemini-based recommendation",
            highlight="Gemini AI pick",
        )


# %% [markdown]
# ## Zero Shot Model Set Up - Please Enter In your Sample Size

# %% [markdown]
# ### Optomized Step 1 - Change from Gemeni to Zero Shot Classifier for moods


# %%
# --- Load Finalized Model (zero-shot) with Caching ---
def load_final_model_data():
    global business_df
    business_df = pd.read_parquet("Data/filtered_restaurants_businesses.parquet")
    review_df = pd.read_parquet("Data/filtered_restaurants_reviews.parquet")
    sample_reviews = review_df.sample(n=10, random_state=42).reset_index(drop=True)

    mood_scores = []
    for review in tqdm(sample_reviews["text"], desc="Scoring reviews"):
        try:
            result = classifier(review, candidate_labels=mood_labels, multi_label=True)
            score_dict = dict(zip(result["labels"], result["scores"]))
        except:
            score_dict = {}
        mood_scores.append(score_dict)

    sample_reviews["mood_scores"] = mood_scores

    def average_mood_scores(df):
        mood_sums = {}
        count = len(df)
        for mood_dict in df["mood_scores"]:
            if isinstance(mood_dict, dict):
                for mood, score in mood_dict.items():
                    mood_sums[mood] = mood_sums.get(mood, 0) + score
        return {mood: mood_sums[mood] / count for mood in mood_sums}

    mood_by_business = (
        sample_reviews.groupby("business_id")
        .apply(average_mood_scores)
        .reset_index(name="aggregated_moods")
    )

    def get_top_mood_and_score(mood_dict):
        if isinstance(mood_dict, dict) and mood_dict:
            top_mood = max(mood_dict.items(), key=lambda x: x[1])
            return pd.Series({"top_mood": top_mood[0], "top_mood_score": top_mood[1]})
        return pd.Series({"top_mood": None, "top_mood_score": None})

    mood_by_business = pd.merge(
        mood_by_business,
        business_df[["business_id", "name", "stars", "review_count"]],
        on="business_id",
        how="left",
    )

    mood_by_business[["top_mood", "top_mood_score"]] = mood_by_business[
        "aggregated_moods"
    ].apply(get_top_mood_and_score)
    mood_by_business["final_score"] = (
        mood_by_business["top_mood_score"]
        * mood_by_business["stars"]
        * mood_by_business["review_count"]
    )
    mood_by_business = mood_by_business.dropna(subset=["top_mood"])
    return mood_by_business


def load_or_generate_final_model_data(overwrite_cache=False):
    if os.path.exists(CACHE_PATH) and not overwrite_cache:
        print("✅ Loading precomputed mood_by_business from cache...")
        return pd.read_parquet(CACHE_PATH)
    else:
        print("⏳ Generating mood_by_business...")
        mood_by_business = load_final_model_data()
        mood_by_business.to_parquet(CACHE_PATH)
        return mood_by_business


# Usage (force refresh with new sample size)
mood_by_business = load_or_generate_final_model_data(overwrite_cache=True)


def get_final_model_recommendation(mood, mood_by_business):
    if mood_by_business is None or mood_by_business.empty:
        print("🚨 mood_by_business is empty!")
        return "No recommendation found."

    top_matches = mood_by_business[mood_by_business["top_mood"] == mood]
    
    if top_matches.empty:
        print(f"🚨 No matches found for mood: {mood}")
        return "No recommendation found."
    
    return top_matches.nlargest(10, "final_score").sample(1).iloc[0]["name"]

# %% [markdown]
# ## Sentence Transformer - Pulling In preprocessed data and loading model

# %% [markdown]
# ### Optomized Step 2 - Zero Shot Classifier to the Better Performing Sentence Transformer - Please note that this model was run in preprocessing to save time and the mood outputs were saved to the dataset in hugging face

# %% [markdown]
# ### Please See Code below gradio output for the accuracy scores

# %%
# --- Tony's Model from Local File ---
from huggingface_hub import hf_hub_download

REPO_ID = "tlockhart/philly_reviews_with_mood.parquet"
FILE_NAME = "philly_reviews_with_mood.parquet"


def load_parquet_from_huggingface(repo_id=REPO_ID, filename=FILE_NAME):
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_parquet(path)


def recommend_restaurant_by_mood_content(df, mood, num_of_recommendations=5):
    mood_matches = df[df["mood"] == mood]
    if mood_matches.empty:
        return None
    mood_expert_id = mood_matches["user_id"].value_counts().idxmax()
    mood_expert_reviews = mood_matches[mood_matches["user_id"] == mood_expert_id].copy()
    mood_expert_reviews["short_review"] = mood_expert_reviews["review"].apply(
        lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x
    )
    mood_expert_reviews = mood_expert_reviews.sort_values(
        by="review_stars", ascending=False
    )
    top_recommendations = mood_expert_reviews.head(num_of_recommendations)
    max_score = top_recommendations["review_stars"].max()
    top_scoring_restaurants = top_recommendations[
        top_recommendations["review_stars"] == max_score
    ]
    final_best = top_scoring_restaurants.sample(
        1, random_state=random.randint(1, 9999)
    ).iloc[0]
    return final_best["business_name"]


# %% [markdown]
# ## Getting Business information from recommendations


# %%
# --- Output Formatter ---
def lookup_business_details(name, mood):
    if not isinstance(name, str) or name.strip() == "" or name == "No recommendation found.":
        return RestaurantDetail(
            phone="Unknown",
            address="Unknown",
            summary="No summary available",
            moods=mood,
            highlight="N/A",
        )

    match = business_df[business_df["name"].str.lower() == name.lower()]

    if not match.empty:
        row = match.iloc[0]
        dynamic_highlight = (
            f"Rated {row['stars']} stars with {row['review_count']} reviews."
        )
        return RestaurantDetail(
            phone="N/A",
            address=f"{row['address']}, {row['city']}, {row['state']} {row['postal_code']}",
            summary=f"{', '.join(row['categories'].split(','))} spot.",
            moods=mood,
            highlight=dynamic_highlight,
        )
    details = fetch_restaurant_details(name)
    if details.address == "Unknown":
        return RestaurantDetail(
            phone="Unknown",
            address="(Generated) Philadelphia",
            summary="Recommended by Gemini for the mood",
            moods="Gemini-based recommendation",
            highlight="Gemini AI pick",
        )
    return details


# %% [markdown]
# ## UI APP LOGIC

# %%
# --- UI + App Logic ---


# --- Display Formatter ---
def format_output(name, details):
    return f"""
🍽️ {name}
📍 Address: {details.address}
📝 Summary: {details.summary}
🎯 Moods: {details.moods}
✨ Highlight: {details.highlight}
    """.strip()


def get_all_recommendations(mood):
    print(f"👉 Selected mood: {mood}")
    print(f"👉 mood_by_business head: \n{mood_by_business.head()}")

    name1 = get_final_model_recommendation(mood, mood_by_business)
    name2 = recommend_restaurant_by_mood_content(tony_reviews, mood)
    name3 = get_llm_generated_recommendation(mood)

    # Fallback if no name is found
    if not name1 or name1 == "No recommendation found.":
        name1 = "No good match found."
    if not name2 or name2 == "No recommendation found.":
        name2 = "No good match found."
    if not name3 or name3 == "No recommendation found.":
        name3 = "No good match found."

    # Now format safely
    out1 = format_output(name1, lookup_business_details(name1, mood))
    out2 = format_output(name2, lookup_business_details(name2, mood))
    out3 = format_output(name3, lookup_business_details(name3, mood))

    return out1, out2, out3


# %% [markdown]
# ## Actual Gradio Set Up

# %%
# --- Gradio UI ---
# --- Local image paths for moods ---
# --- UI creation ---
# --- Local image paths for moods ---
mood_images = {
    "adventurous": "Pictures/adventurous.png",
    "comforting": "Pictures/comforting.png",
    "cozy": "Pictures/cozy.png",
    "energizing": "Pictures/energizing.png",
    "festive": "Pictures/festive.png",
    "indulgent": "Pictures/indulgent.png",
    "refreshing": "Pictures/refreshing.png",
    "romantic": "Pictures/romantic.png",
}
mood_items = [(img_path, mood) for mood, img_path in mood_images.items()]
# --- Map image path back to mood ---
# def mood_from_path(selection):
#     logging.info(f"SELECTION: {selection}")
#     if isinstance(selection, (list, tuple)):
#         selection = selection[0]
#     if isinstance(selection, (list, tuple)):
#         selection = selection[0]

#     for mood, img_path in mood_images.items():
#         if os.path.normpath(selection) == os.path.normpath(img_path):
#             return mood
#     return "adventurous"
# def mood_from_path(selection):
#     logging.info(f"SELECTION: {selection}")
#     if isinstance(selection, (list, tuple)):
#         selection = selection[0]
#     if isinstance(selection, (list, tuple)):
#         selection = selection[0]


#     for mood, img_path in mood_images.items():
#         if os.path.normpath(selection) == os.path.normpath(img_path):
#             return mood
#     return "adventurous"
def mood_from_path(selection):
    logging.info(f"SELECTION: {selection}")
    for mood, img_path in mood_images.items():
        if os.path.normpath(selection) == os.path.normpath(img_path):
            return mood
    return "adventurous"  # Default if not found


# --- Trigger all models ---
# def handle_mood_click(image_path):
#     mood = mood_from_path(image_path)
#     print(f"🧠 Mood selected: {mood}")
#     return get_all_recommendations(mood)
# def handle_mood_click(selected_item):
#     image_path, mood = selected_item  # unpack the tuple
#     print(f"🧠 Mood selected: {mood} (image: {image_path})")
#     return get_all_recommendations(mood)
# def handle_mood_click(selected_image_path):
#     print(f"🖼️ Raw selected item: {selected_image_path}")
#     mood = mood_from_path(selected_image_path)
#     print(f"🧠 Mood selected: {mood}")
#     return get_all_recommendations(mood)
def handle_mood_click(evt: gr.SelectData):
    print(f"🖼️ Raw selected item: {evt.value}")  # <- evt.value contains the selected image path
    mood = mood_from_path(evt.value)
    print(f"🧠 Mood selected: {mood}")
    return get_all_recommendations(mood)


# --- Gradio UI creation ---
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown(
            "## How do you feel today?\nClick one of the moods below to get your restaurant recommendations."
        )

        # --- Gallery without preview_size ---
        gallery = gr.Gallery(
            value=mood_items,
            label="Select Your Mood",
            columns=4,
            rows=2,
            object_fit="cover",
            allow_preview=False,
            show_label=False,
            height=700,  # This keeps layout in view
        )

        # --- Shared output boxes below ---
        out1 = gr.Textbox(label="Zeroshot - bart-large-mnli")
        out2 = gr.Textbox(label="Sentence Transformer - all-MiniLM-L6-v2")
        out3 = gr.Textbox(label="Base Model - Gemeni Recommendation")

        # --- Hook image click to recommendation function ---
        gallery.select(
            fn=handle_mood_click, outputs=[out1, out2, out3]
        )

    return interface


# --- Launch ---
# --- Load Data and Launch ---
mood_by_business = load_or_generate_final_model_data()
tony_reviews = load_parquet_from_huggingface(REPO_ID, FILE_NAME)
demo = create_interface()
demo.launch()


# %% [markdown]
# ### THE OPTOMIZATION CHECK WE DID TO UTILIZE SENTENCE TRANSFORMER -- Performs better than zero shot

# %%
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Sample 10 reviews and their labeled moods
reviews = [
    ("The dim lighting and soft music made it perfect for a date night.", "romantic"),
    ("After a long day, this place just felt like a warm hug.", "comforting"),
    ("Every dish had a spicy kick—totally fired me up!", "energizing"),
    (
        "We wore sweaters, had hot chocolate, and watched the snowfall from inside.",
        "cozy",
    ),
    (
        "Twinkling lights and Christmas songs everywhere—like a holiday dream.",
        "festive",
    ),
    ("The desserts were over-the-top and totally worth every bite.", "indulgent"),
    (
        "We hiked first, then found this open-air cafe with mountain views.",
        "refreshing",
    ),
    (
        "Live jazz, old-school cocktails, and candlelight—it felt like time travel.",
        "romantic",
    ),
    ("The staff gave warm blankets and tea on a rainy night.", "comforting"),
    ("We danced under the stars after margaritas—total vacation vibes!", "energizing"),
]

true_moods = [label for _, label in reviews]
texts = [text for text, _ in reviews]

# Mood options
mood_labels = [
    "adventurous",
    "comforting",
    "energizing",
    "romantic",
    "cozy",
    "festive",
    "indulgent",
    "refreshing",
]

# Load models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Zero-shot predictions ---
zero_shot_preds = []
for text in texts:
    result = classifier(text, candidate_labels=mood_labels)
    zero_shot_preds.append(result["labels"][0])  # top prediction

# --- Embedding-based predictions ---
# Embed all review texts and mood labels
review_embeddings = embedder.encode(texts, convert_to_tensor=True)
mood_embeddings = embedder.encode(mood_labels, convert_to_tensor=True)

embedding_preds = []
for review_emb in review_embeddings:
    sims = util.pytorch_cos_sim(review_emb, mood_embeddings)[0]
    best_idx = sims.argmax().item()
    embedding_preds.append(mood_labels[best_idx])

# --- Accuracy ---
zero_shot_acc = accuracy_score(true_moods, zero_shot_preds)
embedding_acc = accuracy_score(true_moods, embedding_preds)

# --- Print results ---
print("\nEvaluation Results:\n")
for i, text in enumerate(texts):
    print(f"Review {i+1}: {text}")
    print(f"  True Mood        : {true_moods[i]}")
    print(f"  Zero-Shot Predict: {zero_shot_preds[i]}")
    print(f"  Embedding Predict: {embedding_preds[i]}\n")

print("📊 Accuracy Scores")
print(f"Zero-Shot Model Accuracy    : {zero_shot_acc:.2f}")
print(f"Sentence Embedding Accuracy : {embedding_acc:.2f}")
