# 📍 Mood-Based Restaurant Recommendation System

Welcome to our final machine learning project: a **Mood-Based Restaurant Recommendation System** that intelligently recommends Philadelphia restaurants based on how the user is feeling. This end-to-end solution combines **NLP**, **transformers**, **sentence embeddings**, **zero-shot classification**, and **generative AI** to deliver emotionally resonant dining suggestions.

---

## 🧠 Project Overview
This project leverages modern machine learning and NLP technologies to map user moods to restaurant recommendations. It uses:

- **Zero-shot classification** to interpret review sentiment.
- **Sentence embeddings** to compute mood similarity.
- **Google's Gemini API** to generate human-like, mood-aligned restaurant suggestions.
- **DALL·E** to create visuals representing restaurants.
- **Gradio** for a user-friendly front-end.

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone git@github.com:tlockhart/project-3.git
cd project-3
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate   # For Windows
```

### 3. Install Requirements
Install all necessary libraries:
```bash
pip install torch
pip install pandas
pip install transformers
pip install sentence-transformers
pip install gradio
pip install python-dotenv
pip install openai
pip install google-generativeai
pip install langchain-google-genai==2.1.1
pip install langchain 
pip install scikit-learn
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY=your_google_generative_ai_key
```

You can also manually enter the keys in the notebook/code using:
```python
import google.generativeai as genai
genai.configure(api_key="your_google_key_here")
```

### 5. Required Downloads
Make sure the following models/downloads are available:
- `facebook/bart-large-mnli` from Hugging Face for zero-shot classification
- `all-MiniLM-L6-v2` via `sentence-transformers`
- Gemini via `google.generativeai`
- DALL·E via `openai.Image.create()`

---

### 6. Program Execution
1. Open the `Complete Notebook.ipynb` Jupyter Notebook. 
2. Click "Run All" in the Visual Studio Code or Colab. 
3. If prompted for a kernel, select Python 3.10.4.1 and `venv`virtual environment created above.  
4. Click the URL link to open in your browser 
5. Program should execute as shown below  

---

### 7. Execution Demo
[![Watch the video](https://img.icons8.com/ios-filled/100/000000/play-button-circled.png)](https://youtu.be/GvQZsyW6raE)
---

## 💡 Technologies Used

| Tool / Library            | Purpose                                 |
|---------------------------|-----------------------------------------|
| `transformers` (HuggingFace) | Zero-shot mood classification         |
| `sentence-transformers`   | Embedding-based similarity analysis     |
| `LangChain`               | LLM interaction and prompting           |
| `Google Generative AI (Gemini)` | AI-generated restaurant suggestions |
| `Gradio`                  | Frontend web interface                  |
| `DALL·E`                  | Mood-specific image generation          |
| `scikit-learn`            | Accuracy scoring                        |
| `pandas`, `torch`         | Data handling and model support         |

---

## 🧪 Model Evaluation

Throughout the project, we iterated through **three different model strategies**, each adding more nuance and control:

1. **Gemini-only Recommendation**: Initially, we used Google’s Gemini to generate a recommendation directly from the user's mood. While powerful and simple, it provided no explainability or customization.

2. **Zero-Shot Classification**: We added a layer of interpretability by analyzing the sentiment or mood expressed in user reviews using the `facebook/bart-large-mnli` model. This helped us match review content to moods without training a new model.

3. **Sentence Embeddings**: To enhance control and fine-tune the matching, we used `sentence-transformers` to compute cosine similarity between moods and reviews. This gave us more precise control over how closely a review matched a target mood.

We decided to include all three methods in the final product so **users can choose the method that best fits their use case**—be it accuracy, speed, or explainability.

To assess model performance, we manually labeled 10 restaurant-style review texts with true mood labels. We then compared the predicted mood from two models:

### 🔹 Zero-Shot Classification (`facebook/bart-large-mnli`)
This model predicts mood using zero-shot learning. 

You can adjust the number of reviews by changing the `reviews` list in the evaluation block. The top-1 label from `result["labels"]` is used.

### 🔹 Embedding Similarity (`all-MiniLM-L6-v2`)
We used cosine similarity between review embeddings and predefined mood label embeddings.

### 📊 Results

| Model               | Accuracy |
|---------------------|----------|
| Zero-Shot Model     | 0.40     |
| Embedding Model     | 0.50     |

Predictions were logged in a Pandas DataFrame and visualized for clarity.

---

## 🌆 Sample Output
- Example mood: `cozy`
- Gemini-generated suggestion: *"Talula's Daily"*
- DALL·E-generated image: A warm, candle-lit cafe with rustic wooden tables and bookshelves

_**Note**: All images used were created via OpenAI's DALL·E model._

---

## 📈 Application Flow

1. User selects a mood.
2. The system finds restaurants that match this mood using:
   - Embedding similarity
   - Zero-shot classification of review text
   - Gemini for new, real recommendations
3. Images of recommended restaurants are generated using DALL·E.
4. The results are presented in a web app built with Gradio.

---

## 🛠️ Future Enhancements

- Deploy the app on Hugging Face Spaces or AWS
- Include Yelp API for real-time restaurant data
- Use Whisper for voice-to-mood interpretation
- Add multilingual mood prompts
- Expand datasets to include global restaurants
- Add additional options (i.e. food allergies)
- Enhance personalization features by adding users location data

---

## 🧑‍🤝‍🧑 Group Members
- Ryan Bryan [Data Science Lead, UI Integrator]
- Yujing Li [Frontend & Deployment]
- Daniel Liu [NLP & LLM Developer]
- Tony Lockhart [Semantic Classification]
- Cathy Schassberger [Presentation]

---

## 📂 Repo Structure
```
.
├── app.py                     # Gradio interface
├── Complete Notebook.ipynb   # Jupyter notebook with model logic
├── mood_examples.csv         # Manually labeled sample reviews
├── .env                      # API keys (not committed)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

---

## ✅ Grading Checklist
- [x] Jupyter notebook with model development and cleaning
- [x] Python scripts and API integrations
- [x] Model evaluation with accuracy scoring
- [x] One external library not used in class (DALL·E)
- [x] Clean GitHub structure with `.gitignore`
- [x] Presentation
- [x] Presentation-ready README

---

> Built with love, creativity, and transformers 🤖 in Spring 2025
