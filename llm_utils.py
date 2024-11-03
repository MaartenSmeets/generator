# llm_utils.py

import hashlib
import json
import logging
import requests
import re

logger = logging.getLogger(__name__)

def generate_cache_key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()

def send_llm_request(prompt, cache, model_name, api_url, expect_json=True):
    cache_key = generate_cache_key(prompt + model_name)
    if cache_key in cache:
        logger.info("Cache hit for LLM prompt.")
        return cache[cache_key]

    payload = {"model": model_name, "prompt": prompt}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, json=payload, headers=headers, stream=True)

        # Collect streaming response
        response_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    # Parse each line as JSON to check done status
                    line_json = json.loads(line.decode('utf-8'))
                    if "response" in line_json:
                        response_text += line_json["response"]
                    if line_json.get("done", False):
                        break
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON line in response stream.")
                    continue

        # Once complete, attempt to parse accumulated response
        logger.debug(f"Complete LLM response: {response_text}")

        if expect_json:
            # Use regex to isolate JSON content in case there is surrounding text
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                llm_response = json.loads(json_content)  # Parse JSON content
                cache[cache_key] = llm_response  # Save to persistent cache
                return llm_response
            else:
                logger.error("Failed to extract JSON from accumulated response.")
                return {}  # Default to empty dict if JSON not found
        else:
            cache[cache_key] = response_text.strip()
            return response_text.strip()

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request failed: {e}")
        return "" if not expect_json else {}  # Return empty response on error
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response from accumulated response.")
        return ""
