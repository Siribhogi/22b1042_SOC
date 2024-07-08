# LLMs.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import random
import os

# Load environmental variables from .env file
load_dotenv()

model1=ChatGoogleGenerativeAI(model="gemini-pro")
model2=ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
model3=ChatGroq(temperature=0, model_name="llama3-8b-8192")
model4=ChatGroq(temperature=0, model_name="gemma2-9b-it")
model5=ChatGroq(temperature=0, model_name="llama3-70b-8192")
model6=ChatCohere(model="command")

model_dict={
    "gemini-pro": model1,
    "mixtral-8x7b-32768": model2,
    "llama3-8b-8192": model3,
    "gemma2-9b-it": model4,
    "llama3-70b-8192": model5,
    "command": model6
}

def get_random_models(number: int = 2):
    return random.sample(list(model_dict.values()),k=number)

def get_response_model(model, prompt):
    response = model.invoke(prompt)
    return response.content