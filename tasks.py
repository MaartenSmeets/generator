# tasks.py
from typing import List, Dict, Any, Union, Callable
import requests
import os
import json
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import pytesseract
from PIL import Image
from googletrans import Translator
from rake_nltk import Rake
import subprocess
from llm_utils import send_llm_request
import logging
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import hf_hub_download
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

logger = logging.getLogger(__name__)

# Download NLTK resources automatically if they are missing
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Load spaCy model, downloading it if necessary
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

OLLAMA_URL = "http://localhost:11434/api/generate"  # Replace with your actual endpoint
SUMMARIZER_MODEL_NAME = "llama3.1:8b-instruct-fp16"  # Configurable summarizer model name

# Define a dictionary to map task names to functions
TASK_FUNCTIONS: Dict[str, Callable[[Dict[str, Any], Any], Dict[str, Any]]] = {}

def list_tasks() -> List[Dict[str, Any]]:
    """Return a list of available tasks with descriptions, parameters, and outcomes."""
    return [
        {
            "name": "search_query",
            "description": "Perform a search query to gather relevant information.",
            "parameters": ["query"],
            "outcomes": ["search_results"]
        },
        {
            "name": "extract_content",
            "description": "Extract relevant content from a webpage.",
            "parameters": ["url"],
            "outcomes": ["page_content"]
        },
        {
            "name": "validate_fact",
            "description": "Validate the truthfulness of a statement using fact-checking.",
            "parameters": ["statement"],
            "outcomes": ["is_true", "confidence", "sources"]
        },
        {
            "name": "summarize_text",
            "description": "Generate a summary of the provided text.",
            "parameters": ["text"],
            "outcomes": ["summary"]
        },
        {
            "name": "analyze_sentiment",
            "description": "Analyze the sentiment of the provided text.",
            "parameters": ["text"],
            "outcomes": ["sentiment", "confidence"]
        },
        {
            "name": "extract_entities",
            "description": "Extract named entities from text.",
            "parameters": ["text"],
            "outcomes": ["entities"]
        },
        {
            "name": "answer_question",
            "description": "Answer a question based on provided context.",
            "parameters": ["question", "context"],
            "outcomes": ["answer"]
        },
        {
            "name": "extract_text_from_image",
            "description": "Extract text from an image.",
            "parameters": ["image_path"],
            "outcomes": ["extracted_text"]
        },
        {
            "name": "translate_text",
            "description": "Translate text to a specified language.",
            "parameters": ["text", "language"],
            "outcomes": ["translated_text"]
        },
        {
            "name": "parse_json",
            "description": "Parse relevant data from a JSON file.",
            "parameters": ["file_path"],
            "outcomes": ["parsed_data"]
        },
        {
            "name": "extract_keywords",
            "description": "Extract keywords or topics from the provided text.",
            "parameters": ["text"],
            "outcomes": ["keywords"]
        }
    ]

# Task function implementations
def search_query(task_params, cache=None):
    query = task_params.get('query')
    # Read API key from local file
    try:
        with open('serper_api_key.txt', 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        return {"error": "API key file not found"}
    url = 'https://google.serper.dev/search'
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        'q': query
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        results = response.json()
        # Process the results as needed
        # For simplicity, let's extract the organic results
        search_results = []
        for item in results.get('organic', []):
            search_results.append({
                'url': item.get('link'),
                'title': item.get('title'),
                'summary': item.get('snippet')
            })
        return {"search_results": search_results}
    else:
        return {"error": "Failed to retrieve search results"}
TASK_FUNCTIONS["search_query"] = search_query

def extract_content(task_params, cache=None):
    url = task_params.get('url')
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Capture a screenshot of the webpage
        screenshot = capture_screenshot(url)

        # Load the screenshot image
        image = Image.open(BytesIO(screenshot))

        # Download and load the OmniParser model and processor
        model_repo = "microsoft/OmniParser"
        processor = BlipProcessor.from_pretrained(model_repo)
        model = BlipForConditionalGeneration.from_pretrained(model_repo)

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")

        # Generate structured data
        outputs = model.generate(**inputs)
        structured_data = processor.decode(outputs[0], skip_special_tokens=True)

        return {'structured_data': structured_data}
    except requests.RequestException as e:
        return {'error': str(e)}
TASK_FUNCTIONS["extract_content"] = extract_content

def capture_screenshot(url):
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")  # Set initial window size

    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Navigate to the specified URL
        driver.get(url)

        # Allow time for the page to load completely
        time.sleep(2)  # Adjust sleep time as necessary

        # Calculate the total height of the page
        total_height = driver.execute_script("return document.body.scrollHeight")

        # Set the window size to the total height of the page
        driver.set_window_size(1920, total_height)

        # Allow time for the window size adjustment
        time.sleep(5)  # Adjust sleep time as necessary

        # Capture the screenshot
        screenshot = driver.get_screenshot_as_png()

        return screenshot
    finally:
        # Close the WebDriver
        driver.quit()

def validate_fact(task_params, cache=None):
    statement = task_params.get('statement')
    if not statement:
        return {"error": "No statement provided for validation."}

    # Step 1: Perform a search query to find relevant pages
    search_results = search_query({'query': statement})
    if 'error' in search_results:
        return {"error": "Failed to retrieve search results for fact validation"}

    # Step 2: Extract content from each search result
    verified_sources = []
    for result in search_results.get('search_results', []):
        url = result.get('url')
        content_result = extract_content({'url': url})

        if 'page_content' in content_result:
            page_content = content_result['page_content'].lower()
            if statement.lower() in page_content:
                verified_sources.append(url)

    # Step 3: Calculate confidence based on verified sources
    is_true = bool(verified_sources)
    confidence = min(1.0, 0.2 + len(verified_sources) * 0.2)  # Confidence increases with more sources

    return {
        "is_true": is_true,
        "confidence": confidence,
        "sources": verified_sources
    }

TASK_FUNCTIONS["validate_fact"] = validate_fact

def summarize_text(task_params, cache=None):
    text = task_params.get('text', '')
    if not text:
        return {"error": "No text provided for summarization."}

    prompt = (
        "Please provide a concise and comprehensive summary of the following text. "
        "Ensure that the summary captures the key points and main ideas in a clear and coherent manner.\n\n"
        f"Text:\n{text}\n\nSummary:"
    )

    summary = send_llm_request(prompt, cache, SUMMARIZER_MODEL_NAME, OLLAMA_URL, expect_json=False)
    return {"summary": summary}

TASK_FUNCTIONS["summarize_text"] = summarize_text

def analyze_sentiment(task_params, cache=None):
    text = task_params.get('text')
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    # Determine the sentiment
    compound = sentiment_scores['compound']
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    confidence = abs(compound)
    return {"sentiment": sentiment, "confidence": confidence}
TASK_FUNCTIONS["analyze_sentiment"] = analyze_sentiment

def extract_entities(task_params, cache=None):
    text = task_params.get('text')
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"type": ent.label_, "entity": ent.text})
    return {"entities": entities}
TASK_FUNCTIONS["extract_entities"] = extract_entities

def answer_question(task_params, cache=None):
    question = task_params.get('question')
    context = task_params.get('context')
    if not question or not context:
        return {"error": "Both 'question' and 'context' must be provided."}

    prompt = (
        "Based on the following context, please provide a detailed and accurate answer to the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    answer = send_llm_request(prompt, cache, SUMMARIZER_MODEL_NAME, OLLAMA_URL, expect_json=False)
    return {"answer": answer}

TASK_FUNCTIONS["answer_question"] = answer_question

def extract_text_from_image(task_params, cache=None):
    image_path = task_params.get('image_path')
    # Ensure that tesseract is installed and configured properly
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        return {"extracted_text": extracted_text}
    except Exception as e:
        return {"error": str(e)}
TASK_FUNCTIONS["extract_text_from_image"] = extract_text_from_image

def translate_text(task_params, cache=None):
    text = task_params.get('text')
    language = task_params.get('language')
    translator = Translator()
    translation = translator.translate(text, dest=language)
    return {"translated_text": translation.text}
TASK_FUNCTIONS["translate_text"] = translate_text

def parse_json(task_params, cache=None):
    file_path = task_params.get('file_path')
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {"parsed_data": data}
    except Exception as e:
        return {"error": str(e)}
TASK_FUNCTIONS["parse_json"] = parse_json

def extract_keywords(task_params, cache=None):
    text = task_params.get('text')
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    keywords = rake_nltk_var.get_ranked_phrases()
    return {"keywords": keywords}
TASK_FUNCTIONS["extract_keywords"] = extract_keywords
