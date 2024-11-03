import os
import requests
import shelve
import hashlib
import logging
import json

# Define constants
OLLAMA_URL = "http://localhost:11434/api/generate"  # Updated to use generate endpoint
MODEL_NAME = "gemma2:9b-instruct-q8_0"  # Replace with your actual model name
OUTPUT_DIR = 'output'  # Output directory for logs and database files
CACHE_FILE = os.path.join(OUTPUT_DIR, 'llm_cache')
LOG_FILE = os.path.join(OUTPUT_DIR, 'app.log')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging to file and console
logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Initialize cache
def init_cache():
    return shelve.open(CACHE_FILE, writeback=False)

# Close cache on exit
cache = init_cache()
def close_cache():
    cache.close()
import atexit
atexit.register(close_cache)

# Generate a unique cache key
def generate_cache_key(prompt):
    key = f"{prompt}"
    return hashlib.sha256(key.encode()).hexdigest()

# Send a request to the LLM using the generate endpoint and yield the response progressively
def send_llm_request(prompt):
    cache_key = generate_cache_key(prompt)
    if cache_key in cache:
        logger.info("Cache hit for prompt.")
        yield cache[cache_key]
        return
    
    # Payload for Ollama's generate API endpoint
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": True}  # Enable streaming
    headers = {"Content-Type": "application/json"}

    full_response = ""

    try:
        # Use streaming to handle the response line by line
        with requests.post(OLLAMA_URL, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Process each line as it's received
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse each JSON line
                        json_data = json.loads(line.decode('utf-8'))
                        if 'response' in json_data:
                            # Append to the full response
                            chunk = json_data['response']
                            full_response += chunk
                            yield chunk  # Yield each chunk as it is received
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")

            # Cache the full response after streaming completes
            cache[cache_key] = full_response
            logger.info("Successfully received full response from LLM and cached it.")

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request failed: {e}")
        yield "Error: Unable to process request."

# Wrapper function to collect the entire response as a single string
def get_llm_response(prompt):
    response_generator = send_llm_request(prompt)
    response = "".join(response_generator)  # Collect all chunks into a single string
    return response

# Example usage
if __name__ == "__main__":
    prompt = "Hello, how can I help you today?"

    logger.info("Starting the LLM request example.")
    response = get_llm_response(prompt)
    logger.info(f"LLM Response: {response}")
    print("LLM Response:", response)
