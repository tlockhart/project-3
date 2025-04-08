# Restaurant Recommendation System
## Overview

<p style="font-size:18px;">This system utilizes sentiment analysis and zero-shot mood classification to recommend
restaurants based on user-selected dining moods. It's powered by machine learning, natural
language processing and interactive visualizations.

## Table of Contents
<ol style="font-size:18px; font-style:italic;">
  <li><a href="#Import Libraries">Import Libraries</a></li>
  <li><a href="#Load Datasets">Load Datasets</a></li>
  <li><a href="#Mood Classification">Mood Classification</a></li>
  <li><a href="#Review Analysis">Review Analysis</a></li>
  <li><a href="#Top Restaurant Moods">Top Restaurant Moods</a></li>
  <li><a href="#Recommendations">Recommendations</a></li>
  <li><a href="#Test Pipeline">Test Pipeline</a></li>
  <li><a href="#Launch UI">Launch the UI</a></li>
  <li><a href="#Reference">Reference</a></li>
</ol>

## Import Libraries
- 'Pandas' For data manipulation and analysis
- 'Gradio' For user interface
- 'Matplotlib' For static plotting of data
- from langchain_google_genai import ChatGoogleGenerativeAI
- from dotenv import load_dotenv, find_dotenv
- from langchain import PromptTemplate
- from langchain.output_parsers import PydanticOutputParser
- from pydantic import Field, BaseModel
-import os
- Hugging Face - zero-shot classification
- pyarrow

## Load Datasets
- Datasets:
    - Business Data: Includes restaurant names, categories and locations
    - Review Data: Contains customer reviews and ratings
- Loading Method: load parquet from a URL

## Pre-trained Zero-shot Model
- A pre-trained language model from HuggingFace

## Mood Classification
The goal of this step is to categorize customer reviews into predefined modds that help
tailor the recommendations.
-   Classify customer reviews using Zero-shot classifications from HuggingFace
Tranformers assigning the following mood labels: Adventurous, Comforting, Energizing,
Romantic, Cozy, Festive, Indulgent, Refreshing.
- Reviews are processed through the model, which assigns the likelihood of each review falling into one of the 8 mood categories.
- Each review is tagged with the mood label that has the highest probability score. These labeled reviews are then aggregated for analysis and recommendation.

## Review Analysis
- Analyze mood distribution with:
    - Value counts for totals
    - Bar plots for visual breakdowns

## Identify Top Restaurants for Moods
- Filter Philadelphia-based restauratns
- Apply content-based filetering with weighted scoring
    - User Reviews
    - Mood relevance
    - Ratings

