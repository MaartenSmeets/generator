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
from translate import Translator
from rake_nltk import Rake
import subprocess
from llm_utils import send_llm_request
import logging
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import hf_hub_download
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)
from urllib.parse import urlparse
import re

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure the output directory exists
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# Create handlers if they don't already exist
if not logger.handlers:
    # Configure the file handler to use the created directory
    file_handler = logging.FileHandler(os.path.join(output_dir, 'tasks.log'))    
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

device = 'cpu' #LLM claims GPU
logger.debug("Loading SOM model.")
som_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
som_model.to(device)
logger.debug("Loading caption model processor.")
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)

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
            "description": (
                "Perform a search query to gather relevant information, with an optional 'from_date' parameter "
                "to specify the time window for the search results (only options are 'past month' and 'past year')"
            ),
            "parameters": ["query", "from_date (optional)"],
            "outcomes": ["search_results"]
        },
        {
            "name": "extract_content",
            "description": "Extract relevant content from a webpage.",
            "parameters": ["url"],
            "outcomes": ["structured_data"]
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
    from_date = task_params.get('from_date', 'past month')
    logger.debug(f"Performing search query: {query}, from_date: {from_date}")
    if not query:
        logger.error("No query provided for search.")
        return {"error": "No query provided for search."}
    # Read API key from local file
    try:
        with open('serper_api_key.txt', 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        logger.error("API key file not found.")
        return {"error": "API key file not found"}
    url = 'https://google.serper.dev/search'
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        'q': query
    }
    # Map 'from_date' to 'tbs' parameter
    tbs_mapping = {
        'past month': 'qdr:m',
        'past year': 'qdr:y',
    }
    tbs_value = tbs_mapping.get(from_date.lower(), 'qdr:m')  # Default to 'past month'
    data['tbs'] = tbs_value
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
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
        logger.debug(f"Search results obtained: {search_results}")
        return {"search_results": search_results}
    except requests.RequestException as e:
        logger.exception("Failed to retrieve search results.")
        return {"error": f"Failed to retrieve search results: {str(e)}"}
    except Exception as e:
        logger.exception("An unexpected error occurred during search query.")
        return {"error": f"An unexpected error occurred: {str(e)}"}
        
TASK_FUNCTIONS["search_query"] = search_query

def extract_content(task_params, cache=None):
    import hashlib
    url = task_params.get('url')
    logger.debug(f"Starting extract_content with URL: {url}")
    if not url:
        logger.error("No URL provided for content extraction.")
        return {"error": "No URL provided for content extraction."}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logger.debug(f"Successfully fetched URL: {url}")

        # Capture a screenshot of the webpage
        screenshot = capture_screenshot(url)
        logger.debug("Captured screenshot of the webpage.")

        # Generate a unique filename using a hash of the URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        # Create a hash of the URL to use as filename
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()

        filename = f"{domain}_{url_hash}.png"
        logger.debug(f"Generated filename for image: {filename}")

        # Define the output directory
        output_dir = os.path.join('output', 'images')
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory: {output_dir}")

        # Define the full path to save the image
        img_path = os.path.join(output_dir, filename)
        logger.debug(f"Full image path: {img_path}")

        # Save the screenshot
        with open(img_path, 'wb') as f:
            f.write(screenshot)
        logger.debug("Screenshot saved.")

        draw_bbox_config = {
            'text_scale': 0.8,
            'text_thickness': 2,
            'text_padding': 3,
            'thickness': 3,
        }
        BOX_THRESHOLD = 0.03

        logger.debug("Starting OCR processing.")
        try:
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                image_path=img_path,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                use_paddleocr=True
            )
            if ocr_bbox_rslt is None:
                logger.error("OCR bbox result is None.")
                raise ValueError("OCR bbox result is None.")
            text, ocr_bbox = ocr_bbox_rslt
            logger.debug(f"OCR text length: {len(text)}, number of OCR boxes: {len(ocr_bbox)}")

            logger.debug("Starting SOM label extraction.")
            som_result = get_som_labeled_img(
                img_path=img_path,
                model=som_model,
                BOX_TRESHOLD=BOX_THRESHOLD,
                output_coord_in_ratio=False,
                ocr_bbox=ocr_bbox,  # Pass the actual OCR bounding boxes
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,  # Pass the extracted text
                use_local_semantics=True,
                iou_threshold=0.1
            )
            if som_result is None:
                logger.error("SOM labeled image result is None.")
                raise ValueError("SOM labeled image result is None.")
            dino_labeled_img, label_coordinates, parsed_content_list = som_result
            logger.debug(f"Parsed content list length: {len(parsed_content_list)}")

            return {'structured_data': parsed_content_list}
        except Exception as e:
            logger.exception("An error occurred during OCR and SOM processing. Falling back to BeautifulSoup.")
            # If OmniParser fails, use BeautifulSoup to extract content
            logger.debug("Using BeautifulSoup to extract content.")
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text content
            texts = soup.stripped_strings
            text_content = ' '.join(texts)
            logger.debug(f"Extracted text content length: {len(text_content)}")
            return {'structured_data': text_content}

    except requests.RequestException as e:
        logger.exception(f"RequestException occurred while fetching URL: {url}")
        return {'error': f"Failed to fetch URL: {str(e)}"}
    except Exception as e:
        logger.exception("An error occurred in extract_content.")
        return {'error': f"An unexpected error occurred: {str(e)}"}

TASK_FUNCTIONS["extract_content"] = extract_content

def capture_screenshot(url):
    logger.debug(f"Capturing screenshot for URL: {url}")
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
        logger.debug("Navigated to URL.")

        # Allow time for the page to load completely
        time.sleep(3)  # Adjust sleep time as necessary

        # Calculate the total height of the page
        total_height = driver.execute_script("return document.body.scrollHeight")
        logger.debug(f"Total page height: {total_height}")

        # Set a maximum height to avoid OpenCV errors
        MAX_HEIGHT = 32766  # SHRT_MAX - 1

        if total_height > MAX_HEIGHT:
            logger.debug(f"Total height {total_height} exceeds maximum height {MAX_HEIGHT}. Limiting to {MAX_HEIGHT}.")
            total_height = MAX_HEIGHT

        # Set the window size to the total height of the page or the maximum height
        driver.set_window_size(1920, total_height)
        logger.debug("Window size set for full page height.")

        # Allow time for the window size adjustment
        time.sleep(3)  # Adjust sleep time as necessary

        # Capture the screenshot
        screenshot = driver.get_screenshot_as_png()
        logger.debug("Screenshot captured.")

        return screenshot
    except Exception as e:
        logger.exception("An error occurred while capturing screenshot.")
        raise
    finally:
        # Close the WebDriver
        driver.quit()
        logger.debug("WebDriver closed.")

def validate_fact(task_params, cache=None):
    statement = task_params.get('statement')
    logger.debug(f"Validating fact: {statement}")
    if not statement:
        logger.error("No statement provided for validation.")
        return {"error": "No statement provided for validation."}

    # Step 1: Perform a search query to find relevant pages
    search_results = search_query({'query': statement})
    if 'error' in search_results:
        logger.error(f"Failed to retrieve search results: {search_results['error']}")
        return {"error": f"Failed to retrieve search results: {search_results['error']}"}

    # Step 2: Extract content from each search result
    verified_sources = []
    for result in search_results.get('search_results', []):
        url = result.get('url')
        content_result = extract_content({'url': url})

        if 'structured_data' in content_result:
            page_content = content_result['structured_data']
            page_content_str = json.dumps(page_content).lower()
            if statement.lower() in page_content_str:
                verified_sources.append(url)
        elif 'error' in content_result:
            logger.error(f"Error extracting content from URL {url}: {content_result['error']}")

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
    logger.debug("Summarizing text.")
    if not text:
        logger.error("No text provided for summarization.")
        return {"error": "No text provided for summarization."}

    prompt = (
        "Provide a summary of the following text, focusing only on the main points and key information. "
        "Do not include any introductions, comments about the summarization process, or closing remarks. "
        "If the content is messy, focus on the information you can obtain, considering the page is likely focused on a single topic.\n\n"
        f"{text}\n\nSummary:"
    )

    try:
        summary = send_llm_request(prompt, cache, SUMMARIZER_MODEL_NAME, OLLAMA_URL, expect_json=False)
        return {"summary": summary.strip()}
    except Exception as e:
        logger.exception("An error occurred while summarizing text.")
        return {"error": f"An error occurred while summarizing text: {str(e)}"}

TASK_FUNCTIONS["summarize_text"] = summarize_text

def analyze_sentiment(task_params, cache=None):
    text = task_params.get('text')
    logger.debug("Analyzing sentiment.")
    if not text:
        logger.error("No text provided for sentiment analysis.")
        return {"error": "No text provided for sentiment analysis."}
    try:
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
    except Exception as e:
        logger.exception("An error occurred while analyzing sentiment.")
        return {"error": f"An error occurred while analyzing sentiment: {str(e)}"}
TASK_FUNCTIONS["analyze_sentiment"] = analyze_sentiment

def extract_entities(task_params, cache=None):
    text = task_params.get('text')
    logger.debug("Extracting entities.")
    if not text:
        logger.error("No text provided for entity extraction.")
        return {"error": "No text provided for entity extraction."}
    try:
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({"type": ent.label_, "entity": ent.text})
        return {"entities": entities}
    except Exception as e:
        logger.exception("An error occurred while extracting entities.")
        return {"error": f"An error occurred while extracting entities: {str(e)}"}
TASK_FUNCTIONS["extract_entities"] = extract_entities

def answer_question(task_params, cache=None):
    question = task_params.get('question')
    context = task_params.get('context')
    logger.debug(f"Answering question: {question}")
    if not question or not context:
        logger.error("Both 'question' and 'context' must be provided.")
        return {"error": "Both 'question' and 'context' must be provided."}

    prompt = (
        "Based on the following context, please provide a detailed and accurate answer to the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    try:
        answer = send_llm_request(prompt, cache, SUMMARIZER_MODEL_NAME, OLLAMA_URL, expect_json=False)
        return {"answer": answer}
    except Exception as e:
        logger.exception("An error occurred while answering question.")
        return {"error": f"An error occurred while answering question: {str(e)}"}

TASK_FUNCTIONS["answer_question"] = answer_question

def extract_text_from_image(task_params, cache=None):
    image_path = task_params.get('image_path')
    logger.debug(f"Extracting text from image: {image_path}")
    if not image_path:
        logger.error("No image path provided for text extraction.")
        return {"error": "No image path provided for text extraction."}
    # Ensure that tesseract is installed and configured properly
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        return {"extracted_text": extracted_text}
    except Exception as e:
        logger.exception("An error occurred while extracting text from image.")
        return {"error": f"An error occurred while extracting text from image: {str(e)}"}
TASK_FUNCTIONS["extract_text_from_image"] = extract_text_from_image

def translate_text(task_params, cache=None):
    text = task_params.get('text')
    language = task_params.get('language')
    logger.debug(f"Translating text to {language}.")

    if not text:
        logger.error("No text provided for translation.")
        return {"error": "No text provided for translation."}
    if not language:
        logger.error("No target language specified for translation.")
        return {"error": "No target language specified for translation."}

    try:
        translator = Translator(to_lang=language)
        translated_text = translator.translate(text)
        return {"translated_text": translated_text}
    except Exception as e:
        logger.exception("An error occurred during translation.")
        return {"error": f"Translation failed: {str(e)}"}

TASK_FUNCTIONS["translate_text"] = translate_text

def parse_json(task_params, cache=None):
    file_path = task_params.get('file_path')
    logger.debug(f"Parsing JSON file: {file_path}")
    if not file_path:
        logger.error("No file path provided for JSON parsing.")
        return {"error": "No file path provided for JSON parsing."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return {"parsed_data": data}
    except Exception as e:
        logger.exception("An error occurred while parsing JSON.")
        return {"error": f"An error occurred while parsing JSON: {str(e)}"}
TASK_FUNCTIONS["parse_json"] = parse_json

def extract_keywords(task_params, cache=None):
    text = task_params.get('text')
    logger.debug("Extracting keywords.")
    if not text:
        logger.error("No text provided for keyword extraction.")
        return {"error": "No text provided for keyword extraction."}
    try:
        rake_nltk_var = Rake()
        rake_nltk_var.extract_keywords_from_text(text)
        keywords = rake_nltk_var.get_ranked_phrases()
        return {"keywords": keywords}
    except Exception as e:
        logger.exception("An error occurred while extracting keywords.")
        return {"error": f"An error occurred while extracting keywords: {str(e)}"}
TASK_FUNCTIONS["extract_keywords"] = extract_keywords

