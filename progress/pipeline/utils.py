# utils.py

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel
from transformers import pipeline
import pandas as pd
from huggingface_hub import hf_hub_download
import os
import random

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError(".env file not found!")

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REPO_ID = os.getenv("REPO_ID")
FILE_NAME = os.getenv("FILE_NAME")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Check your .env file.")

# Define output schema
class restaurant_detail(BaseModel):
    phone: str
    address: str
    summary: str
    moods: str
    highlight: str
    rating: str
    hours: str
    price: str
    popular_items: str

def setup_prompt_template(query):
    format = """You are a world famous restaurant expert. ...
    Question: {query}
    Answer: ... (format)
    """
    return PromptTemplate(input_variables=["query"], template=format)

def get_details_from_llm(restaurant_name, restaurant_city, restaurant_street):
    query = f"Give me the details of {restaurant_name} in {restaurant_city} on {restaurant_street}"
    prompt_template = setup_prompt_template(query)
    parser = PydanticOutputParser(pydantic_object=restaurant_detail)
    instructions = parser.get_format_instructions()
    query += "\n\n" + instructions
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0.3)
    response = (prompt_template | llm).invoke(query)
    data = parser.parse(response.content)
    return data

def format_restaurant_details(data, mood):
    return f"""**Summary:** {data.summary}
📞 Phone: {data.phone}
📍 Address: {data.address}
🎯 Moods: {mood}
✨ Highlight: {data.highlight}
⭐ Rating: {data.rating}
🕒 Hours: {data.hours}
💵 Price: {data.price}
🔥 Popular Items: {data.popular_items}
"""

def translate(input_text, target_language):
    language_codes = {
        "French": "fra_Latn",
        "German": "deu_Latn",
        "Romanian": "ron_Latn"
    }
    translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", device="mps")
    target_code = language_codes.get(target_language)
    if not target_code:
        raise ValueError(f"Language {target_language} not supported!")

    lines = input_text.split("\n")
    translated_lines = []
    for line in lines:
        if line.strip() == "":
            translated_lines.append("")
        else:
            translated_line = translator(line, src_lang="eng_Latn", tgt_lang=target_code)[0]["translation_text"]
            translated_lines.append(translated_line)

    return "\n".join(translated_lines)

def load_parquet_from_huggingface(repo_id, filename):
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        df = pd.read_parquet(file_path)
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

def recommend_restaurant_by_mood_content(df, mood, num_of_recommendations=5):
    mood_matches = df[df["mood"] == mood]
    if mood_matches.empty:
        return None
    mood_expert_id = mood_matches["user_id"].value_counts().idxmax()
    mood_expert_reviews = mood_matches[mood_matches["user_id"] == mood_expert_id].copy()
    mood_expert_reviews["short_review"] = mood_expert_reviews["review"].apply(
        lambda x: x[:50] + "..." if isinstance(x, str) and len(x) > 50 else x
    )
    mood_expert_reviews = mood_expert_reviews.sort_values(by="review_stars", ascending=False)
    top_recommendations = mood_expert_reviews.head(num_of_recommendations)
    max_score = top_recommendations["review_stars"].max()
    top_scoring_restaurants = top_recommendations[top_recommendations["review_stars"] == max_score]
    final_best = top_scoring_restaurants.sample(1, random_state=random.randint(1, 9999)).iloc[0]
    return final_best