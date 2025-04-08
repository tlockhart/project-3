# ui.py
# Import the following Libraries
"""
conda create -n project3 python=3.10
conda activate project3
pip install langchain==0.3.23
pip install langchain-core==0.3.51
pip install langchain-community==0.3.21
pip install langchain-google-genai==2.1.2
pip install gradio
pip install transformers
pip install transformers==4.36.2
pip install python-dotenv
pip install torch
pip install pyarrow
Requirement: MacOS 12.3+

Create a .env file in the root directory with the following variables:
GEMINI_API_KEY= Enter your Google GEMINI_API_Key
REPO_ID = "tlockhart/philly_reviews_with_mood.parquet"
FILE_NAME = "philly_reviews_with_mood.parquet"

Run Gradio: python gradio_ui.py
"""

import gradio as gr
from utils import load_parquet_from_huggingface, recommend_restaurant_by_mood_content, get_details_from_llm, format_restaurant_details, translate, REPO_ID, FILE_NAME

# Load dataset
customer_reviews_df = load_parquet_from_huggingface(REPO_ID, FILE_NAME)

if customer_reviews_df is None:
    raise ValueError("Dataset failed to load.")

# Mood images (✅ exactly yours)
mood_images = [
    ("Adventurous", "https://thumbs.dreamstime.com/b/cute-pirate-smiley-wearing-black-pirate-scarf-eye-patch-emoticon-emoji-vector-illustration-96990738.jpg"),
    ("Comforting", "https://previews.123rf.com/images/yayayoy/yayayoy1305/yayayoy130500001/19481706-dreamy-emoticon-with-his-head-propped-by-his-hands.jpg"),
    ("Energizing", "https://previews.123rf.com/images/suslo/suslo1304/suslo130400008/18953821-yellow-sign-of-emotion-with-a-mohawk-and-hands.jpg"),
    ("Romantic", "https://previews.123rf.com/images/yayayoy/yayayoy1602/yayayoy160200012/52420163-male-emoticon-blowing-a-kiss.jpg"),
    ("Cozy", "https://previews.123rf.com/images/yayayoy/yayayoy1205/yayayoy120500007/13629388-meditating-emoticon.jpg"),
    ("Festive", "https://thumbs.dreamstime.com/b/christmas-emoticon-bell-27501954.jpg"),
    ("Indulgent", "https://previews.123rf.com/images/yayayoy/yayayoy1210/yayayoy121000009/15836615-emoticon-eating-an-apple.jpg"),
    ("Refreshing", "https://png.pngtree.com/thumb_back/fh260/background/20241030/pngtree-happy-emoji-made-of-water-droplets-refreshing-playful-design-with-reflective-image_16473684.jpg"),
]

def getSuggestion(mood):
    user_selected_mood = mood.lower()
    recommendation = recommend_restaurant_by_mood_content(customer_reviews_df, user_selected_mood)
    if recommendation is None:
        return "No restaurants found for this mood!"
    rec_object = {
        "name": recommendation["business_name"],
        "address": recommendation["address"],
        "city": recommendation["city"]
    }
    restaurant_details = get_details_from_llm(rec_object["name"], rec_object["city"], rec_object["address"])
    return format_restaurant_details(restaurant_details, user_selected_mood.title())

def create_interface():
    with gr.Blocks() as interface:
        selected_mood = gr.State(value="Adventurous")
        
        # ✅ Your custom styles
        gr.HTML("""
        <style>
        #gallery {
            width: fit-content;
        }
        .grid-wrap.fixed-height {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .grid-container {
            height: auto !important;
        }

        .thumbnail-item {
            width: 300px !important;
            height: auto !important;
        }
        .gallery-item img {
            width: 100px;
            height: 100px;
            object-fit: contain;
        }
        .gallery {
            justify-content: center;
        }
        </style>
        """)

        mood_gallery = gr.Gallery(
            label="How do you feel today?",
            elem_id="gallery",
            columns=4,
            allow_preview=False,
            show_label=True
        )

        mood_items = [(img_url, label) for label, img_url in mood_images]

        def load_gallery():
            return mood_items

        interface.load(
            fn=load_gallery,
            inputs=[],
            outputs=[mood_gallery]
        )

        def on_image_select(evt: gr.SelectData):
            clicked_index = evt.index
            selected_label = mood_images[clicked_index][0]
            return selected_label

        mood_gallery.select(
            fn=on_image_select,
            inputs=[],
            outputs=[selected_mood]
        )

        suggestion_button = gr.Button("Find a restaurant")
        suggestion_output = gr.Textbox(label="Suggested activity", interactive=False, visible=False)

        target_language = gr.Dropdown(
            choices=["English", "French", "German", "Romanian"],
            label="Please select a Language",
            visible=False
        )
        language_confirm_button = gr.Button("Translate", visible=False)
        translate_output = gr.Textbox(label="Translated suggestion", interactive=False, visible=False)

        def on_suggestion_button_click(mood):
            suggestion = getSuggestion(mood)
            return (
                gr.update(value=suggestion, visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False)
            )

        def on_translate_button_click(text, language):
            translated_text = translate(text, language)
            return gr.update(value=translated_text, visible=True)

        suggestion_button.click(
            fn=on_suggestion_button_click,
            inputs=[selected_mood],
            outputs=[suggestion_output, target_language, language_confirm_button, translate_output]
        )

        language_confirm_button.click(
            fn=on_translate_button_click,
            inputs=[suggestion_output, target_language],
            outputs=[translate_output]
        )

    return interface

if __name__ == "__main__":
    gradio_ui = create_interface()
    gradio_ui.launch(share=True)