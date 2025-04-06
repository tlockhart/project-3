#!/usr/bin/env python
# coding: utf-8

# # This is a final file for getting the restaurant details from an LLM

# ## Import Libraries

# In[1]:


from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import Field, BaseModel

import os


# ## Setting the LLM Model and API Key

# In[2]:


# Locate .env file, and load environment variables.
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError(".env file not found!")


# Set the model name for our LLMs.
GEMINI_MODEL = "gemini-2.0-flash"

# Load the API key to a variable.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Check your .env file.")


# In[3]:


# Define the class for our parsed responses.
class restaurant_detail(BaseModel):
    phone: str = Field(description="Phone")
    address: str = Field(description="Address")
    summary: str = Field(description="Summary")
    moods: str = Field(description="Moods")
    highlight: str = Field(description="Highlight")
    rating: str = Field(description="Rating")
    hours: str = Field(description="Hours")
    price: str = Field(description="Price")
    popular_items: str = Field(description="Popular Items")


# ## Set up prompt tempplate

# In[4]:


def setup_prompt_template(query):
    # Define the format for the template.
    format = """You are a world famous restaurant expert.  Answer only questions that would be about restaurants.
    If the human asks questions not related to restaurant, remind them that your job is to help them get the details of a restaurnt
    Question: {query}
    Answer:

        **Your response should have the following information:**

        **Summary:** [Summary of the restaurant]
        **Phone:** [Phone number of the restaurant]
        **Address:** [Address of the restaurent]
        **Moods:** [Moods of the restaurant]
        **Highlight:** [quality of food, service, value]
        **Rating:** [Rating]
        **Hours:** [hours of business]
        **Price:** [usual price range per person]
        **Popular Items:** [popluar menu items]

        For example:
        **Summary:** Chama Gaucha is a high-end churrascaria (Brazilian steakhouse) offering a wide selection of grilled meats carved tableside by gauchos (Brazilian cowboys)
        **Phone:** (713) 357-9777
        **Address:** 5655 Westheimer Rd Suite A, Houston, TX 77056
        **Moods:** Vibrant, lively, and celebratory
        **Highlight:** Reviewers consistently praise the "perfectly cooked meats" and the "robust salad bar" with fresh options. 
                    The service is frequently described as "impeccable" and "attentive," with staff ensuring glasses are filled and meat preferences are met.
        **Rating:** 4.7
        **Hours:**  Sunday - Thursday: 11 AM - 10 PM
                    Friday - Saturday: 11 AM - 11 PM
        **Price:**  $30 - $50
        **Popular Items:** Picanha, Churrasco, Salad Bar

        Do not include any extra text or formatting. 
    """

    # Construct the prompt template.
    prompt_template = PromptTemplate(
        input_variables=["query"], 
        template = format)
    
    return prompt_template


# ## Define the final entry function

# In[5]:


def get_details_from_llm(restaurant_name, restaurant_city, restaurant_street):
    
    query = f"Give me the details of {restaurant_name} in {restaurant_city} on {restaurant_street}"

    prompt_template = setup_prompt_template(query)

    # Initialize the output parser using the schema.
    parser = PydanticOutputParser(pydantic_object = restaurant_detail)

    # Get the output format instructions and print them.
    instructions = parser.get_format_instructions()

    # Define a query as a string, combining with the instructions.
    query += "\n\n" + instructions

    # Create the LangChain Model
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY, temperature=0.3)

    # Pass the query to the invoke method, and print the result.  
    response = (prompt_template | llm).invoke(query)

    # Parse the result, store it, and print it.
    data = parser.parse(response.content)

    return data
