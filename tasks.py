from typing import List, Dict, Any, Union, Callable
import requests
import os
import json
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline, AutoTokenizer
import pytesseract
from PIL import Image
from googletrans import Translator
from rake_nltk import Rake
import subprocess
import torch

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

# Initialize the summarizer and tokenizer once
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


# Define a dictionary to map task names to functions
TASK_FUNCTIONS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

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
def search_query(task_params):
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

def extract_content(task_params):
    url = task_params.get('url')
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the page
        texts = soup.find_all(text=True)
        page_content = ' '.join(text.strip() for text in texts if text.strip())
        return {'page_content': page_content}
    except requests.RequestException as e:
        return {'error': str(e)}
TASK_FUNCTIONS["extract_content"] = extract_content

def validate_fact(task_params):
    statement = task_params.get('statement')
    # Use search_query function to get search results for the statement
    search_results = search_query({'query': statement})
    if 'error' in search_results:
        return {"error": "Failed to validate fact"}
    # Fetch and parse the pages using the updated extract_content function
    verified_sources = []
    for result in search_results.get('search_results', []):
        content_result = extract_content({'url': result['url']})
        if 'page_content' in content_result and 'error' not in content_result:
            if statement.lower() in content_result['page_content'].lower():
                verified_sources.append(result['url'])
    is_true = bool(verified_sources)
    confidence = 0.9 if is_true else 0.1
    sources = verified_sources
    return {"is_true": is_true, "confidence": confidence, "sources": sources}
TASK_FUNCTIONS["validate_fact"] = validate_fact

def summarize_text(task_params):
    text = task_params.get('text', '')
    if not text:
        return {"error": "No text provided for summarization."}

    max_input_length = tokenizer.model_max_length  # 1024 tokens for BART
    inputs = tokenizer.encode(text, return_tensors='pt')
    input_length = inputs.shape[1]

    if input_length > max_input_length:
        # Split the text into chunks
        chunks = []
        stride = max_input_length - 200  # Overlap for better context
        for i in range(0, input_length, stride):
            end = i + max_input_length
            chunk_input_ids = inputs[:, i:end]
            chunk_text = tokenizer.decode(chunk_input_ids[0], skip_special_tokens=True)
            chunks.append(chunk_text)
    else:
        chunks = [text]

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    final_summary = ' '.join(summaries)
    return {"summary": final_summary}

TASK_FUNCTIONS["summarize_text"] = summarize_text

def analyze_sentiment(task_params):
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

def extract_entities(task_params):
    text = task_params.get('text')
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"type": ent.label_, "entity": ent.text})
    return {"entities": entities}
TASK_FUNCTIONS["extract_entities"] = extract_entities

def answer_question(task_params):
    question = task_params.get('question')
    context = task_params.get('context')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    answer = qa_pipeline(question=question, context=context)
    return {"answer": answer['answer']}
TASK_FUNCTIONS["answer_question"] = answer_question

def extract_text_from_image(task_params):
    image_path = task_params.get('image_path')
    # Ensure that tesseract is installed and configured properly
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        return {"extracted_text": extracted_text}
    except Exception as e:
        return {"error": str(e)}
TASK_FUNCTIONS["extract_text_from_image"] = extract_text_from_image

def translate_text(task_params):
    text = task_params.get('text')
    language = task_params.get('language')
    translator = Translator()
    translation = translator.translate(text, dest=language)
    return {"translated_text": translation.text}
TASK_FUNCTIONS["translate_text"] = translate_text

def parse_json(task_params):
    file_path = task_params.get('file_path')
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {"parsed_data": data}
    except Exception as e:
        return {"error": str(e)}
TASK_FUNCTIONS["parse_json"] = parse_json

def extract_keywords(task_params):
    text = task_params.get('text')
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    keywords = rake_nltk_var.get_ranked_phrases()
    return {"keywords": keywords}
TASK_FUNCTIONS["extract_keywords"] = extract_keywords
