# tasks.py

from typing import List, Dict, Any, Callable, Optional, Tuple
import json
import logging
import os
import io
import re
import subprocess
import time
import hashlib
from urllib.parse import urlparse
import selenium
import requests
from bs4 import BeautifulSoup
from lxml import html, etree
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
import spacy
from PIL import Image
import pytesseract
from translate import Translator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from llm_utils import send_llm_request
from utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model,
)

# -------------------- Logging Configuration --------------------

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

# -------------------- Initialization of Models and Resources --------------------

device = 'cpu'  # Adjust as necessary
logger.debug("Loading YOLO model.")
som_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
som_model.to(device)
logger.debug("Loading caption model processor.")
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=device
)

# Define the resources and their specific paths
nltk_resources = {
    "stopwords": "corpora/stopwords",
    "punkt": "tokenizers/punkt"
}

# Download NLTK resources automatically if they are missing
for resource_name, resource_path in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
        logger.debug(f"NLTK resource '{resource_name}' is already downloaded.")
    except LookupError:
        logger.debug(f"Downloading NLTK resource: '{resource_name}'")
        nltk.download(resource_name, quiet=True)
        # Verify download
        try:
            nltk.data.find(resource_path)
            logger.debug(f"Successfully downloaded NLTK resource: '{resource_name}'")
        except LookupError:
            logger.error(f"Failed to download NLTK resource: '{resource_name}'")

# Load spaCy model, downloading it if necessary
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.debug("Downloading spaCy model: en_core_web_sm")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# -------------------- Constants --------------------

OLLAMA_URL = "http://localhost:11434/api/generate"  # Replace with your actual endpoint
TASK_GENERATION_MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Configurable model name for task generation
SUMMARIZER_MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Configurable summarizer model name
IS_REPUTABLE_SOURCE_MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Replace with your desired model name
IS_CONTENT_VALUABLE_MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Configurable model name for content validation
EVALUATE_QUESTION_FOCUS_MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Configurable model name for question focus evaluation
EVALUATE_SUMMARY_MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Configurable model name for summary evaluation
VALIDATE_ANSWER_MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Configurable model name for answer validation
GENERATE_SUBQUESTIONS_MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"  # Configurable model name for subquestion generation

# -------------------- Helper Functions --------------------

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the first JSON object found in the response text.
    """
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            logger.error("No JSON object found in the response.")
            return None
        json_str = response_text[json_start:json_end]
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        logger.exception(f"JSON decoding failed: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during JSON extraction: {e}")
        return None

def enforce_json_response(description: str, expected_keys: List[str]) -> str:
    """
    Constructs a prompt that enforces the LLM to return a JSON response with specific keys.
    """
    prompt = (
        f"{description}\n\n"
        "Return ONLY a JSON object with the following keys: " + ", ".join([f"'{key}'" for key in expected_keys]) + ". "
        "Do not include any additional text, explanations, or code blocks. "
        "Ensure the JSON is properly formatted."
    )
    return prompt

# -------------------- Task Definitions --------------------

def list_tasks() -> List[Dict[str, Any]]:
    """Return a list of available public tasks with descriptions, parameters, and outcomes."""
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
            "parameters": ["url", "question"],
            "outcomes": ["extracted_text", "method_used", "score"]
        },
        {
            "name": "validate_fact",
            "description": "Validate the truthfulness of a statement using fact-checking.",
            "parameters": ["statement"],
            "outcomes": ["is_true", "confidence", "sources"]
        },
        {
            "name": "is_reputable_source",
            "description": "Determine if a URL is from a reputable source using LLM.",
            "parameters": ["url"],
            "outcomes": ["is_reputable"]
        },
        {
            "name": "summarize_text",
            "description": "Generate a summary of the provided text.",
            "parameters": ["text", "question"],
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
        },
        {
            "name": "evaluate_question_focus",
            "description": "Evaluate whether a question is focused and detailed on a single topic.",
            "parameters": ["question"],
            "outcomes": ["is_focused"]
        },
        {
            "name": "evaluate_summary",
            "description": "Evaluate whether a summary is valuable for inclusion in the final answer.",
            "parameters": ["summary", "question"],
            "outcomes": ["is_valuable"]
        },
        {
            "name": "generate_subquestions",
            "description": "Generate subquestions based on the main question.",
            "parameters": ["question"],
            "outcomes": ["subquestions"]
        },
        {
            "name": "validate_answer",
            "description": "Determine if the provided answer indicates that the question cannot be answered based on the supplied context.",
            "parameters": ["question", "answer"],
            "outcomes": ["is_unanswerable"]
        }
    ]

# Define a dictionary to map **public** task names to functions
TASK_FUNCTIONS_PUBLIC: Dict[str, Callable[[Dict[str, Any], Any], Dict[str, Any]]] = {}

# -------------------- Public Task Function Implementations --------------------

def search_query(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
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

def extract_content(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    """
    Extract relevant content from a webpage using multiple methods and select the most valuable output based on text length.

    Args:
        task_params (Dict[str, Any]): Parameters containing 'url' and 'question'.
        cache (Any): Optional cache parameter.

    Returns:
        Dict[str, Any]: Extracted content or error information.
    """
    url = task_params.get('url')
    question = task_params.get('question')
    logger.debug(f"Starting extract_content with URL: {url}")
    if not url:
        logger.error("No URL provided for content extraction.")
        return {"error": "No URL provided for content extraction."}
    if not question:
        logger.error("No question provided for content extraction.")
        return {'error': "No question provided for content extraction."}

    # Generate a unique hash for the URL to ensure unique filenames
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('.', '_')  # Replace dots to avoid issues in filenames

    # Define the output directory for images and extracted texts
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    logger.debug(f"Images and extracted texts directory: {images_dir}")

    # Check if the URL is a YouTube link
    is_youtube = False
    if 'youtube.com' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc:
        is_youtube = True

    extracted_texts = {}
    scores = {}

    try:
        if is_youtube:
            logger.debug("Detected YouTube URL. Attempting to fetch transcript.")
            transcript_result = _fetch_youtube_transcript(url)
            if 'transcript' in transcript_result:
                method_name = 'Method_YouTube_Transcript'
                extracted_texts[method_name] = transcript_result['transcript']
                logger.debug("Transcript obtained successfully.")

                # Write the extracted transcript to a text file
                filename = f"{domain}_{url_hash}_{method_name}.txt"
                filename = re.sub(r'\W+', '_', filename)  # Sanitize filename
                file_path = os.path.join(images_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(transcript_result['transcript'])
                logger.debug(f"Transcript written to {file_path}")
            else:
                logger.error(f"Failed to fetch transcript: {transcript_result.get('error')}")
        else:
            # Capture screenshot and page source once using Selenium/UC
            logger.debug("Capturing screenshot and obtaining page source using Selenium/UC.")
            screenshot, page_source = _capture_screenshot(url)
            if not page_source:
                logger.error("Failed to obtain page source using Selenium/UC.")
                extracted_texts['Method_1_Selenium_UC'] = None
            else:
                # Method 1: Use page_source
                try:
                    method_name = 'Method_1_Selenium_UC'
                    cleaned_html = _clean_html_content(page_source)
                    text_content = _extract_text_from_html(cleaned_html)
                    text_content = _cleanup_extracted_text(text_content)
                    extracted_texts[method_name] = text_content
                    logger.debug("Method 1: Cleaned and extracted text from Selenium/UC page source.")

                    # Write the extracted text to a text file
                    filename = f"{domain}_{url_hash}_{method_name}.txt"
                    filename = re.sub(r'\W+', '_', filename)  # Sanitize filename
                    file_path = os.path.join(images_dir, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    logger.debug(f"Extracted text from Method 1 written to {file_path}")
                except Exception as e:
                    logger.exception(f"Method 1 failed: {e}")
                    extracted_texts['Method_1_Selenium_UC'] = None

            # Method 2: Perform OCR with SOM labeling using the screenshot
            if screenshot:
                try:
                    logger.debug("Method 2: Performing OCR with SOM labeling on the screenshot.")
                    # Save the screenshot
                    filename_first_part = f"{domain}_{url_hash}"
                    filename_first_part = re.sub(r'\W+', '_', filename_first_part)  # Sanitize filename
                    filename = f"{filename_first_part}_Method_2_Check_OCR_Box.png"
                    img_path = os.path.join(images_dir, filename)
                    with open(img_path, 'wb') as f:
                        f.write(screenshot)
                    logger.debug("Method 2: Screenshot saved.")

                    # Perform OCR and SOM labeling
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
                            logger.error("Method 2: OCR bbox result is None.")
                            extracted_texts['Method_2_Check_OCR_Box'] = None
                        else:
                            text, ocr_bbox = ocr_bbox_rslt
                            logger.debug(f"Method 2: OCR text length: {len(text)}, number of OCR boxes: {len(ocr_bbox)}")

                            som_result = get_som_labeled_img(
                                img_path=img_path,
                                model=som_model,
                                BOX_TRESHOLD=0.03,  # Example threshold
                                output_coord_in_ratio=False,
                                ocr_bbox=ocr_bbox,  # Pass the actual OCR bounding boxes
                                draw_bbox_config={
                                    'text_scale': 0.8,
                                    'text_thickness': 2,
                                    'text_padding': 3,
                                    'thickness': 3,
                                },
                                caption_model_processor=caption_model_processor,
                                ocr_text=text,  # Pass the extracted text
                                use_local_semantics=True,
                                iou_threshold=0.1
                            )
                            if som_result is None:
                                logger.error("Method 2: SOM labeled image result is None.")
                                extracted_texts['Method_2_SOM_Labeling'] = None
                            else:
                                dino_labeled_img, label_coordinates, parsed_content_list = som_result
                                method_label = 'Method_2_SOM_Labeling'
                                logger.debug(f"Method 2: Parsed content length: {len(parsed_content_list)}")
                                cleaned_extract = _clean_som_output(parsed_content_list)
                                extracted_texts[method_label] = cleaned_extract

                                logger.debug(f"Method 2: Cleaned parsed content length: {len(cleaned_extract)}")

                                # Write the extracted content to a text file
                                filename = f"{domain}_{url_hash}_{method_label}.txt"
                                filename = re.sub(r'\W+', '_', filename)  # Sanitize filename
                                file_path = os.path.join(images_dir, filename)
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(cleaned_extract)
                                logger.debug(f"Extracted content from Method 2 written to {file_path}")
                    except Exception as e:
                        logger.exception(f"Method 2: OCR processing failed: {e}")
                        extracted_texts['Method_2_Check_OCR_Box'] = None
                except Exception as e:
                    logger.exception(f"Method 2 failed: {e}")
                    extracted_texts['Method_2_Check_OCR_Box'] = None
            else:
                logger.error("Screenshot not available for Methods 2 and 3.")
                extracted_texts['Method_2_Check_OCR_Box'] = None
                extracted_texts['Method_3_Pytesseract'] = None

            # Method 3: Use pytesseract to OCR the screenshot
            if screenshot:
                try:
                    logger.debug("Method 3: Using pytesseract for OCR on screenshot.")
                    extracted_text_pytesseract = _extract_text_with_pytesseract(screenshot)
                    method_name = 'Method_3_Pytesseract'
                    if extracted_text_pytesseract:
                        extracted_texts[method_name] = extracted_text_pytesseract
                        logger.debug("Method 3: Pytesseract OCR successful.")

                        # Write the extracted text to a text file
                        filename = f"{domain}_{url_hash}_{method_name}.txt"
                        filename = re.sub(r'\W+', '_', filename)  # Sanitize filename
                        file_path = os.path.join(images_dir, filename)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(extracted_text_pytesseract)
                        logger.debug(f"Extracted text from Method 3 written to {file_path}")
                    else:
                        logger.error("Method 3: Pytesseract returned empty text.")
                        extracted_texts[method_name] = None
                except Exception as e:
                    logger.exception(f"Method 3 failed: {e}")
                    extracted_texts['Method_3_Pytesseract'] = None
            else:
                extracted_texts['Method_3_Pytesseract'] = None

            # Method 4: Use requests to fetch the page and clean it up
            try:
                logger.debug("Method 4: Fetching and cleaning content using requests.")
                page_source_content_result = _extract_page_source_content({"url": url}, cache)
                method_name = 'Method_4_Requests'
                if 'extracted_text' in page_source_content_result:
                    extracted_texts[method_name] = page_source_content_result['extracted_text']
                    logger.debug("Method 4: Fetched and cleaned content using requests.")

                    # Write the extracted text to a text file
                    filename = f"{domain}_{url_hash}_{method_name}.txt"
                    filename = re.sub(r'\W+', '_', filename)  # Sanitize filename
                    file_path = os.path.join(images_dir, filename)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(page_source_content_result['extracted_text'])
                    logger.debug(f"Extracted text from Method 4 written to {file_path}")
                else:
                    logger.error(f"Method 4 failed: {page_source_content_result.get('error')}")
                    extracted_texts[method_name] = None
            except Exception as e:
                logger.exception(f"Method 4 failed: {e}")
                extracted_texts['Method_4_Requests'] = None

        # Now, evaluate each method's output based on the length of the extracted text
        for method, text in extracted_texts.items():
            if text:
                if isinstance(text, str):
                    score = len(text)
                else:
                    # For non-string types, convert to string and get length
                    score = len(str(text))
                scores[method] = score
                logger.debug(f"{method} scored {score} based on extracted text length.")
            else:
                scores[method] = 0
                logger.debug(f"{method} scored 0 due to failed extraction.")

        # Select the method with the highest score
        best_method = max(scores, key=scores.get) if scores else None
        best_score = scores.get(best_method, 0) if best_method else 0

        if best_method and extracted_texts.get(best_method):
            logger.debug(f"Best method: {best_method} with score {best_score}")
            extracted_text = extracted_texts[best_method]
            return {"extracted_text": extracted_text, "method_used": best_method, "score": best_score}
        else:
            logger.error("All extraction methods failed.")
            return {"error": "All extraction methods failed."}
    except Exception as e:
        logger.exception(f"An error occurred during content extraction: {e}")
        return {"error": f"An error occurred during content extraction: {str(e)}"}

def validate_fact(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    statement = task_params.get('statement')
    logger.debug(f"Validating fact: {statement}")
    if not statement:
        logger.error("No statement provided for validation.")
        return {"error": "No statement provided for validation."}

    # Step 1: Extract entities from the statement
    try:
        doc = nlp(statement)
        entities = [ent.text for ent in doc.ents if ent.label_ in ('ORG', 'GPE', 'PERSON', 'PRODUCT')]
        logger.debug(f"Extracted entities: {entities}")
    except Exception as e:
        logger.exception("An error occurred while extracting entities from the statement.")
        entities = []

    # Step 2: Perform a search query to find relevant pages
    search_results = execute_task("search_query", {'query': statement}, cache)
    if 'error' in search_results:
        logger.error(f"Failed to retrieve search results: {search_results['error']}")
        return {"error": f"Failed to retrieve search results: {search_results['error']}"}

    # Step 3: Extract content from each search result
    verified_sources = []
    for result in search_results.get('search_results', []):
        url = result.get('url')

        # Check if the URL is from a reputable source using execute_task
        reputable_result = execute_task("is_reputable_source", {"url": url}, cache)
        if reputable_result.get("is_reputable"):
            content_result = execute_task("extract_content", {'url': url, 'question': statement}, cache)

            if 'extracted_text' in content_result:
                page_content = content_result['extracted_text']
                page_content_str = page_content.lower()
                if statement.lower() in page_content_str:
                    verified_sources.append(url)
            elif 'error' in content_result:
                logger.error(f"Error extracting content from URL {url}: {content_result['error']}")
        else:
            logger.debug(f"Skipping non-reputable source: {url}")

    # Step 4: Calculate confidence based on verified sources
    is_true = bool(verified_sources)
    confidence = min(1.0, 0.2 + len(verified_sources) * 0.2)  # Confidence increases with more sources

    logger.debug(f"Fact validation result - is_true: {is_true}, confidence: {confidence}, sources: {verified_sources}")

    return {
        "is_true": is_true,
        "confidence": confidence,
        "sources": verified_sources
    }

def is_reputable_source(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    url = task_params.get('url')
    if not url:
        logger.error("No URL provided for reputation check.")
        return {"error": "No URL provided for reputation check."}

    logger.debug(f"Checking if URL is from a reputable source: {url}")

    # Construct prompt for LLM enforcing JSON response
    description = (
        "Determine whether the following URL is from a reputable and credible source. "
        "Consider international government sites, major tech companies like Meta, Google, OpenAI, Microsoft, Apple, Nvidia, universities, "
        "and reputable news outlets like BBC as reputable sources. Do not consider sources known for fake news like Fox News as reputable."
    )
    expected_keys = ["is_reputable"]
    prompt = enforce_json_response(description, expected_keys) + f"\n\nURL: {url}"

    try:
        response = send_llm_request(
            prompt,
            cache,
            IS_REPUTABLE_SOURCE_MODEL_NAME,
            OLLAMA_URL,
            expect_json=True  # Expecting a JSON response
        )
        if response is None:
            # Attempt to extract JSON from raw response
            response_text = response if isinstance(response, str) else ""
            response = extract_json_from_response(response_text)
            if response is None:
                return {"error": "Failed to extract JSON from LLM response."}

        # Validate and extract the 'is_reputable' field from the JSON response
        is_reputable = response.get("is_reputable")
        if isinstance(is_reputable, bool):
            logger.debug(f"LLM determined 'is_reputable': {is_reputable}")
            return {"is_reputable": is_reputable}
        else:
            logger.error(f"LLM response missing 'is_reputable' field or invalid format: {response}")
            return {"error": "Invalid LLM response format for reputation check."}
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON response from LLM for reputation check.")
        return {"error": "Failed to parse JSON response from LLM."}
    except Exception as e:
        logger.exception("An error occurred during source reputation check.")
        return {"error": f"An error occurred during source reputation check: {str(e)}"}

def summarize_text(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    text = task_params.get('text', '')
    question = task_params.get('question')
    logger.debug("Summarizing text.")
    if not text:
        logger.error("No text provided for summarization.")
        return {"error": "No text provided for summarization."}
    if not question:
        logger.error("No question provided for summarization.")
        return {"error": "No question provided for summarization."}

    prompt = (
        "Provide a summary of the following text, focusing especially on relevant details for answering the given question. "
        "Be detailed and specific on suppliers, products, capabilities, dates and other key details such as strategy and vision.\n\n"
        "Do not include any introductions, comments about the summarization process, or closing remarks.\n\n"
        f"Question:\n{question}\n\nText:\n{text}\n\nSummary:"
    )

    try:
        summary = send_llm_request(prompt, cache, SUMMARIZER_MODEL_NAME, OLLAMA_URL, expect_json=False)
        logger.debug(f"Generated summary: {summary}")
        return {"summary": summary.strip()}
    except Exception as e:
        logger.exception("An error occurred while summarizing text.")
        return {"error": f"An error occurred while summarizing text: {str(e)}"}

def analyze_sentiment(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
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
        logger.debug(f"Sentiment analysis result: sentiment={sentiment}, confidence={confidence}")
        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        logger.exception("An error occurred while analyzing sentiment.")
        return {"error": f"An error occurred while analyzing sentiment: {str(e)}"}

def extract_entities(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
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
        logger.debug(f"Extracted entities: {entities}")
        return {"entities": entities}
    except Exception as e:
        logger.exception("An error occurred while extracting entities.")
        return {"error": f"An error occurred while extracting entities: {str(e)}"}

def answer_question(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
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
        logger.debug(f"Generated answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.exception("An error occurred while answering question.")
        return {"error": f"An error occurred while answering question: {str(e)}"}

def extract_text_from_image(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    image_path = task_params.get('image_path')
    logger.debug(f"Extracting text from image: {image_path}")
    if not image_path:
        logger.error("No image path provided for text extraction.")
        return {"error": "No image path provided for text extraction."}
    # Ensure that tesseract is installed and configured properly
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        logger.debug(f"Extracted text from image: {extracted_text}")
        return {"extracted_text": extracted_text}
    except Exception as e:
        logger.exception("An error occurred while extracting text from image.")
        return {"error": f"An error occurred while extracting text from image: {str(e)}"}

def translate_text(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
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
        logger.debug(f"Translated text: {translated_text}")
        return {"translated_text": translated_text}
    except Exception as e:
        logger.exception("An error occurred during translation.")
        return {"error": f"Translation failed: {str(e)}"}

def parse_json(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    file_path = task_params.get('file_path')
    logger.debug(f"Parsing JSON file: {file_path}")
    if not file_path:
        logger.error("No file path provided for JSON parsing.")
        return {"error": "No file path provided for JSON parsing."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Parsed JSON data: {data}")
        return {"parsed_data": data}
    except Exception as e:
        logger.exception("An error occurred while parsing JSON.")
        return {"error": f"An error occurred while parsing JSON: {str(e)}"}

def extract_keywords(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    text = task_params.get('text')
    logger.debug("Starting keyword extraction process.")
    
    if not text:
        logger.error("No text provided for keyword extraction.")
        return {"error": "No text provided for keyword extraction."}

    # Step 1: Ensure required NLTK resources are available
    required_resources = ['punkt', 'stopwords']
    for resource in required_resources:
        try:
            resource_path = f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}'
            nltk.data.find(resource_path)
            logger.debug(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            logger.debug(f"Downloading missing NLTK resource: '{resource}'")
            nltk.download(resource, quiet=True)
            try:
                nltk.data.find(resource_path)
                logger.debug(f"Successfully downloaded NLTK resource: '{resource}'")
            except LookupError:
                logger.error(f"Failed to download NLTK resource: '{resource}'")
                return {"error": f"Required NLTK resource '{resource}' is missing and could not be downloaded."}

    # Step 2: Initialize Rake and extract keywords
    try:
        rake_nltk_var = Rake(language='english')
        logger.debug("Rake initialized successfully with language='english'. Extracting keywords from text.")
        
        # Extract keywords
        rake_nltk_var.extract_keywords_from_text(text)
        keywords = rake_nltk_var.get_ranked_phrases()
        logger.debug(f"Extracted keywords: {keywords}")

        return {"keywords": keywords}
    except LookupError as e:
        # Handle specific NLTK lookup errors
        logger.exception("A LookupError occurred during keyword extraction.")
        return {"error": f"A LookupError occurred during keyword extraction: {str(e)}"}
    except Exception as e:
        # Log any other exceptions
        logger.exception("An unexpected error occurred during keyword extraction.")
        return {"error": f"An unexpected error occurred during keyword extraction: {str(e)}"}

def evaluate_question_focus(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    question = task_params.get('question')
    if not question:
        logger.error("No question provided for focus evaluation.")
        return {"error": "No question provided for focus evaluation."}

    logger.debug(f"Evaluating focus of question: {question}")

    # Construct prompt for LLM enforcing JSON response
    description = (
        "Determine whether the following question is broad or specific. "
        "A broad question often covers multiple topics or requires extensive explanation, "
        "while a specific question is narrow and focused on a single topic that can be answered concisely. "
        "If the question seems broad, consider it 'not focused'. Even if the question appears specific, "
        "identify any underlying components that might require further decomposition into sub-questions.\n\n"
        "Examples of focused questions:\n"
        "- 'What are the latest NVIDIA AI products released in 2023?'\n"
        "Examples of questions that are not focused:\n"
        "- 'Write a detailed report on recent advancements in AI.'\n"
        "- 'Explain everything about the AI industry.'\n\n"
        "Return ONLY a JSON object with one key: 'is_focused' (a boolean). "
        "Do not include any additional text, explanations, or code blocks. Ensure the JSON is properly formatted."
    )
    expected_keys = ["is_focused"]
    prompt = enforce_json_response(description, expected_keys) + f"\n\nQuestion:\n{question}"

    try:
        response = send_llm_request(
            prompt,
            cache,
            EVALUATE_QUESTION_FOCUS_MODEL_NAME,
            OLLAMA_URL,
            expect_json=True  # Expecting a JSON response
        )
        if response is None:
            return {"error": "Failed to get response from LLM."}

        # Validate and extract the 'is_focused' field from the JSON response
        is_focused = response.get("is_focused")
        if isinstance(is_focused, bool):
            logger.debug(f"LLM determined 'is_focused': {is_focused}")
            return {"is_focused": is_focused}
        else:
            logger.error(f"LLM response missing 'is_focused' field or invalid format: {response}")
            return {"error": "Invalid LLM response format for question focus evaluation."}
    except Exception as e:
        logger.exception("An error occurred during question focus evaluation.")
        return {"error": f"An error occurred during question focus evaluation: {str(e)}"}
    
def evaluate_summary(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    """
    Evaluate whether a summary is valuable for inclusion in the final answer.

    Args:
        task_params (Dict[str, Any]): Parameters containing 'summary' and 'question'.
        cache (Any): Optional cache parameter.

    Returns:
        Dict[str, Any]: Evaluation result indicating if the summary is valuable.
    """
    summary = task_params.get('summary')
    question = task_params.get('question')
    if not summary or not question:
        logger.error("Both 'summary' and 'question' must be provided for summary evaluation.")
        return {"error": "Both 'summary' and 'question' must be provided for summary evaluation."}

    logger.debug(f"Evaluating summary: {summary} for question: {question}")

    # Construct prompt for LLM enforcing JSON response
    description = (
        "Determine whether the following summary is valuable for inclusion in the context to help answer the given question. "
        "A valuable summary should provide clear insights that directly address the question. "
        "If the only information in the summary is an error message, a security warning, or a cookie policy, it is not valuable. "
        "News relevant to answering the question is. When in doubt, consider it valuable."
    )
    expected_keys = ["is_valuable"]
    prompt = enforce_json_response(description, expected_keys) + f"\n\nQuestion:\n{question}\n\nSummary:\n{summary}"

    try:
        response = send_llm_request(
            prompt,
            cache,
            EVALUATE_SUMMARY_MODEL_NAME,
            OLLAMA_URL,
            expect_json=True  # Expecting a JSON response
        )
        if response is None:
            # Attempt to extract JSON from raw response
            response_text = response if isinstance(response, str) else ""
            response = extract_json_from_response(response_text)
            if response is None:
                return {"error": "Failed to extract JSON from LLM response."}

        # Validate and extract the 'is_valuable' field from the JSON response
        is_valuable = response.get("is_valuable")
        if isinstance(is_valuable, bool):
            logger.debug(f"LLM determined 'is_valuable' for summary: {is_valuable}")
            return {"is_valuable": is_valuable}
        else:
            logger.error(f"LLM response missing 'is_valuable' field or invalid format: {response}")
            return {"error": "Invalid LLM response format for summary evaluation."}
    except json.JSONDecodeError:
        logger.exception("Failed to parse JSON response from LLM for summary evaluation.")
        return {"error": "Failed to parse JSON response from LLM."}
    except Exception as e:
        logger.exception("An error occurred during summary evaluation.")
        return {"error": f"An error occurred during summary evaluation: {str(e)}"}

def generate_subquestions(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    """
    Generate subquestions based on the main question.

    Args:
        task_params (Dict[str, Any]): Parameters containing 'question'.
        cache (Any): Optional cache parameter.

    Returns:
        Dict[str, Any]: Generated subquestions or error information.
    """
    question = task_params.get('question')
    logger.debug("Generating subquestions.")
    if not question:
        logger.error("No question provided for subquestion generation.")
        return {"error": "No question provided for subquestion generation."}

    # Construct prompt for LLM enforcing JSON response with example
    description = (
        "Please analyze the following question and decompose it into multiple smaller, specific sub-questions "
        "that cover all aspects necessary to comprehensively answer the main question. "
        "Even if the question seems focused, think about any underlying components or steps that need to be addressed. "
        "Ensure that the combination of answers to these sub-questions will fully answer the main question.\n\n"
        "Return ONLY a JSON object with one key: 'subquestions', which is a list of strings. "
        "Do not include any additional text, explanations, or code blocks. Ensure the JSON is properly formatted."
    )
    expected_keys = ["subquestions"]
    prompt = enforce_json_response(description, expected_keys) + f"\n\nQuestion:\n{question}"

    try:
        response = send_llm_request(
            prompt,
            cache,
            GENERATE_SUBQUESTIONS_MODEL_NAME,
            OLLAMA_URL,
            expect_json=True  # Expecting a JSON response
        )
        if response is None:
            return {"error": "Failed to get response from LLM."}

        sub_questions = response.get("subquestions", [])
        if not isinstance(sub_questions, list) or not all(isinstance(q, str) for q in sub_questions):
            logger.error("Subquestions are not in the expected format.")
            return {"error": "Subquestions are not in the expected format."}
        if not sub_questions:
            logger.error("Failed to generate sub-questions. The response was empty or improperly formatted.")
        else:
            logger.debug(f"Generated sub-questions: {sub_questions}")
        return {"subquestions": sub_questions}
    except Exception as e:
        logger.exception(f"Error generating subquestions: {e}")
        return {"error": f"An error occurred while generating subquestions: {str(e)}"}
    
def validate_answer(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    """
    Determine if the provided answer indicates that the question cannot be answered based on the supplied context.

    Args:
        task_params (Dict[str, Any]): Parameters containing 'question' and 'answer'.
        cache (Any): Optional cache parameter.

    Returns:
        Dict[str, Any]: Evaluation result indicating if the answer is unanswerable.
    """
    question = task_params.get('question')
    answer = task_params.get('answer')
    logger.debug(f"Validating if answer indicates unanswerable for question: {question}")

    if not question or not answer:
        logger.error("Both 'question' and 'answer' must be provided for answer validation.")
        return {"error": "Both 'question' and 'answer' must be provided for answer validation."}

    # Construct prompt for LLM enforcing JSON response
    description = (
        "Determine whether the following answer explicitly states that the question cannot be answered "
        "based on the supplied context. If the answer provides any relevant information or partially answers the question, "
        "even if not completely comprehensive, consider it answerable.\n\n"
        "Return ONLY a JSON object with one key: 'is_unanswerable' (a boolean). "
        "Do not include any additional text, explanations, or code blocks. Ensure the JSON is properly formatted."
    )
    expected_keys = ["is_unanswerable"]
    prompt = enforce_json_response(description, expected_keys) + f"\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"

    try:
        response = send_llm_request(
            prompt,
            cache,
            VALIDATE_ANSWER_MODEL_NAME,
            OLLAMA_URL,
            expect_json=True  # Expecting a JSON response
        )
        if response is None:
            return {"error": "Failed to get response from LLM."}

        # Validate and extract the 'is_unanswerable' field from the JSON response
        is_unanswerable = response.get("is_unanswerable")
        if isinstance(is_unanswerable, bool):
            logger.debug(f"LLM determined 'is_unanswerable': {is_unanswerable}")
            return {"is_unanswerable": is_unanswerable}
        else:
            logger.error(f"LLM response missing 'is_unanswerable' field or invalid format: {response}")
            return {"error": "Invalid LLM response format for answer validation."}
    except Exception as e:
        logger.exception("An error occurred during answer validation.")
        return {"error": f"An error occurred during answer validation: {str(e)}"}
    
# Register only public tasks
for task in list_tasks():
    task_name = task["name"]
    try:
        TASK_FUNCTIONS_PUBLIC[task_name] = globals()[task_name]
        logger.debug(f"Registered task function: {task_name}")
    except KeyError:
        logger.error(f"Function '{task_name}' is not defined and cannot be registered.")

# -------------------- Private Task Function Implementations --------------------

def _fetch_youtube_transcript(url: str) -> Dict[str, Any]:
    """
    Private helper function to fetch YouTube transcript.
    """
    logger.debug(f"Fetching YouTube transcript for URL: {url}")
    try:
        # Extract the video ID from the URL
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc:
            query_params = dict([part.split('=') for part in parsed_url.query.split('&') if '=' in part])
            video_id = query_params.get('v')
        elif 'youtu.be' in parsed_url.netloc:
            video_id = parsed_url.path.lstrip('/')
        else:
            logger.error("Invalid YouTube URL format.")
            return {"error": "Invalid YouTube URL format."}
        
        if not video_id:
            logger.error("Video ID not found in the URL.")
            return {"error": "Video ID not found in the URL."}
        
        # Fetch the transcript using YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])
        logger.debug("Transcript fetched successfully.")
        return {"transcript": transcript_text}
    except VideoUnavailable:
        logger.error("The video is unavailable.")
        return {"error": "The video is unavailable."}
    except TranscriptsDisabled:
        logger.error("Transcripts are disabled for this video.")
        return {"error": "Transcripts are disabled for this video."}
    except NoTranscriptFound:
        logger.error("No transcript found for this video.")
        return {"error": "No transcript found for this video."}
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching the transcript.")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def _extract_page_source_content(task_params: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    """
    Extract text content from the page source of a URL.

    Args:
        task_params (Dict[str, Any]): Parameters containing 'url'.
        cache (Any): Optional cache parameter.

    Returns:
        Dict[str, Any]: Extracted text or error information.
    """
    url = task_params.get('url')
    logger.debug(f"Fetching page source content from URL: {url}")
    if not url:
        logger.error("No URL provided for page source extraction.")
        return {"error": "No URL provided for page source extraction."}
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        logger.debug("Page fetched successfully via requests.")
        cleaned_html = _clean_html_content(response.content)
        text_content = _extract_text_from_html(cleaned_html)
        text_content = _cleanup_extracted_text(text_content)
        logger.debug(f"Extracted text length: {len(text_content)}")
        return {"extracted_text": text_content}
    except requests.RequestException as e:
        logger.exception(f"Failed to fetch URL via requests: {url}")
        return {"error": f"Failed to fetch URL via requests: {str(e)}"}
    except Exception as e:
        logger.exception(f"An unexpected error occurred during page source extraction: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def _capture_screenshot(url: str, max_retries: int = 3, delay: int = 5) -> Optional[Tuple[bytes, str]]:
    """
    Private helper function to capture a screenshot of the given URL and return the image bytes and page source.
    If it fails after retries, returns (None, None).
    """
    logger.debug(f"Capturing screenshot for URL: {url}")

    # Validate URL
    parsed = urlparse(url)
    if not parsed.scheme:
        logger.debug(f"URL scheme missing, adding 'https://': {url}")
        url = 'https://' + url

    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Use the new headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--ignore-certificate-errors")

    # Initialize the undetected Chrome WebDriver with use_subprocess=True
    try:
        logger.debug("Initializing undetected_chromedriver with subprocess.")
        driver = uc.Chrome(options=chrome_options, use_subprocess=True)
    except Exception as e:
        logger.exception("Failed to initialize undetected_chromedriver with subprocess.")
        return None, None

    try:
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Attempt {attempt} to navigate to URL: {url}")
                driver.get(url)
                logger.debug("Navigated to URL.")

                # Wait for the page to load completely
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                logger.debug("Page loaded successfully.")
                break  # Success, exit loop
            except selenium.common.exceptions.WebDriverException as e:
                logger.warning(f"Attempt {attempt} failed with WebDriverException: {e}")
                if attempt < max_retries:
                    logger.debug(f"Retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts to load the URL failed.")
                    return None, None
            except Exception as e:
                logger.exception(f"An unexpected error occurred during driver.get: {e}")
                return None, None

        # Attempt to click away cookie banners
        try:
            logger.debug("Attempting to click away cookie banners.")
            # Define possible XPaths for cookie buttons
            cookie_button_xpaths = [
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'proceed')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'consent')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'confirm')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'ok')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
                "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'save changes')]",
            ]

            for xpath in cookie_button_xpaths:
                try:
                    buttons = driver.find_elements(By.XPATH, xpath)
                    for button in buttons:
                        if button.is_displayed() and button.is_enabled():
                            logger.debug(f"Found cookie button with XPath: {xpath}, attempting to click.")
                            _simulate_human_mouse_movement(driver, button)
                            WebDriverWait(driver, 5).until(EC.invisibility_of_element(button))
                            logger.debug("Cookie banner handled successfully.")
                            break  # Exit after clicking one button
                except Exception as e:
                    logger.debug(f"Could not click cookie button with XPath: {xpath}. Error: {e}")
            logger.debug("Finished attempting to click cookie banners.")
        except Exception as e:
            logger.debug(f"No cookie pop-up found or could not locate buttons: {e}")

        # Allow time for the page to adjust after clicking banners
        time.sleep(2)

        # Get the page source
        try:
            page_source = driver.page_source
            logger.debug("Page source obtained successfully.")
        except Exception as e:
            logger.exception(f"Failed to get page source: {e}")
            page_source = None

        # Now proceed to capture the screenshot

        # Calculate the total height of the page
        try:
            total_height = driver.execute_script("return document.body.scrollHeight")
            logger.debug(f"Total page height: {total_height}")
        except Exception as e:
            logger.warning(f"Failed to get page height: {e}. Using default height 1080.")
            total_height = 1080

        # Set a maximum height to avoid OpenCV errors
        MAX_HEIGHT = 32766  # SHRT_MAX - 1

        if total_height > MAX_HEIGHT:
            logger.debug(f"Total height {total_height} exceeds maximum height {MAX_HEIGHT}. Limiting to {MAX_HEIGHT}.")
            total_height = MAX_HEIGHT

        # Set the window size to the total height of the page or the maximum height
        try:
            driver.set_window_size(1920, total_height)
            logger.debug("Window size set for full page height.")
        except Exception as e:
            logger.warning(f"Failed to set window size to full page height: {e}. Using default size.")
            try:
                driver.set_window_size(1920, 1080)
                logger.debug("Window size set to default 1920x1080.")
            except Exception as ex:
                logger.exception(f"Failed to set default window size: {ex}")
                # Proceed without setting window size

        # Allow time for the window size adjustment
        time.sleep(3)  # Adjust sleep time as necessary

        # Capture the screenshot
        try:
            screenshot = driver.get_screenshot_as_png()
            logger.debug("Screenshot captured.")
        except Exception as e:
            logger.exception(f"Failed to capture screenshot: {e}")
            screenshot = None

        return screenshot, page_source
    except Exception as e:
        logger.exception(f"An error occurred while capturing screenshot: {e}")
        return None, None
    finally:
        # Close the WebDriver
        try:
            driver.quit()
            logger.debug("WebDriver closed successfully.")
        except Exception as e:
            logger.warning(f"An error occurred while quitting the WebDriver: {e}")

def _extract_text_with_pytesseract(screenshot_bytes: bytes) -> Optional[str]:
    """
    Use pytesseract to extract text from screenshot bytes.

    Args:
        screenshot_bytes (bytes): The screenshot image in bytes.

    Returns:
        Optional[str]: The extracted text or None if extraction fails.
    """
    try:
        logger.debug("Method 3: Starting pytesseract OCR on screenshot.")
        image = Image.open(io.BytesIO(screenshot_bytes))
        extracted_text = pytesseract.image_to_string(image)
        logger.debug("Method 3: Pytesseract OCR completed successfully.")
        return extracted_text.strip()
    except Exception as e:
        logger.exception(f"Method 3: Pytesseract OCR failed: {e}")
        return None

def _clean_som_output(output: list) -> str:
    """
    Cleans the SOM labeling output to remove references to boxes and returns a concatenated string
    with each entry on a new line.

    Args:
        output (list): List of labeled text from SOM labeling.

    Returns:
        str: Cleaned concatenated string with each entry on a new line.
    """
    cleaned_lines = []
    
    # Process each item in the input list
    for item in output:
        # Check if item contains a colon and split if it does
        if ": " in item:
            # Extract the content after the identifier
            cleaned_text = item.split(": ", 1)[-1].strip()
        else:
            # If no colon, keep the whole item if it's not empty
            cleaned_text = item.strip()

        # Add to cleaned lines if not empty
        if cleaned_text:
            cleaned_lines.append(cleaned_text)
    
    # Join all cleaned lines into a single string with newlines
    return "\n".join(cleaned_lines)

def _simulate_human_mouse_movement(driver, element):
    """
    Private helper function to simulate human-like mouse movement to interact with a web element.
    """
    from selenium.webdriver.common.action_chains import ActionChains
    import numpy as np
    import random

    try:
        # Get the location and size of the element
        location = element.location_once_scrolled_into_view
        size = element.size

        # Calculate the center of the element
        end_x = location['x'] + size['width'] / 2
        end_y = location['y'] + size['height'] / 2

        # Starting point (you can set this to the current mouse position if available)
        start_x = random.randint(0, driver.execute_script("return window.innerWidth"))
        start_y = random.randint(0, driver.execute_script("return window.innerHeight"))

        # Generate a list of points simulating a human-like mouse movement
        num_points = random.randint(10, 20)
        x_points = np.linspace(start_x, end_x, num_points) + np.random.normal(0, 5, num_points)
        y_points = np.linspace(start_y, end_y, num_points) + np.random.normal(0, 5, num_points)

        actions = ActionChains(driver)
        actions.move_by_offset(start_x, start_y)

        for x, y in zip(x_points, y_points):
            actions.move_by_offset(x - start_x, y - start_y)
            start_x, start_y = x, y
            # Random small delay between movements
            time.sleep(random.uniform(0.01, 0.05))

        # Randomized delay before clicking
        time.sleep(random.uniform(0.5, 1.5))
        actions.click(on_element=element)
        actions.perform()
        logger.debug("Simulated human mouse movement and clicked the element.")
    except Exception as e:
        logger.exception("An error occurred during mouse movement simulation.")
        # If simulation fails, proceed without it
        pass

def _clean_html_content(html_content: bytes) -> str:
    """
    Private helper function to clean the HTML content by removing unnecessary tags and elements.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
        element.extract()

    for element in soup.find_all(attrs={"class": ["sidebar", "advertisement", "promo", "footer", "header"]}):
        element.extract()
    for element in soup.find_all(attrs={"id": ["sidebar", "advertisement", "promo", "footer", "header"]}):
        element.extract()

    cleaned_html = soup.prettify()
    try:
        tree = html.fromstring(cleaned_html)
        for element in tree.xpath('//script|//style|//header|//footer|//nav|//aside|//form'):
            element.drop_tree()

        return etree.tostring(tree, method='html', encoding='unicode')
    except Exception as e:
        logger.error(f"Error in cleaning HTML: {e}")
        return ""

def _extract_text_from_html(cleaned_html: str) -> str:
    """
    Private helper function to extract text from cleaned HTML content.
    """
    soup = BeautifulSoup(cleaned_html, 'html.parser')
    return soup.get_text(separator='\n')

def _cleanup_extracted_text(text: str) -> str:
    """
    Private helper function to cleanup the extracted text by removing excessive whitespace and formatting.
    """
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = '\n'.join([line.strip() for line in text.split('\n')])
    text = re.sub(r'\s+', ' ', text)
    return text

# -------------------- Execute Task Function --------------------

def execute_task(task_name: str, parameters: Dict[str, Any], cache: Any = None) -> Dict[str, Any]:
    """
    Execute a public task by its name with the given parameters.

    Args:
        task_name (str): The name of the task to execute.
        parameters (Dict[str, Any]): Parameters required for the task.
        cache (Any, optional): Optional cache object. Defaults to None.

    Returns:
        Dict[str, Any]: The outcome of the task, including any sub-task outcomes.
    """
    task_function = TASK_FUNCTIONS_PUBLIC.get(task_name)

    # Log task_name and parameters to debug
    logger.debug(f"Executing task: {task_name}, with parameters: {parameters}")
    if not isinstance(parameters, dict):
        logger.error(f"Expected parameters to be a dictionary, got {type(parameters)}")
        raise TypeError(f"Expected parameters to be a dictionary, got {type(parameters)}")

    if task_function:
        try:
            return task_function(parameters, cache)
        except Exception as e:
            logger.exception(f"Error executing task '{task_name}': {e}")
            return {"error": str(e)}
    else:
        logger.error(f"No function found for task: {task_name}")
        raise ValueError(f"No function found for task: {task_name}")
