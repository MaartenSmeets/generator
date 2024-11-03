# generator.py
import sqlite3
import json
import os
import logging
import requests
import hashlib
import re
import shelve
import tasks
from typing import List, Dict, Any, Union, Callable
from llm_utils import send_llm_request

# Define constants and output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"  # Replace with your actual endpoint
MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"  # Replace with your actual model name
DATABASE_FILE = os.path.join(OUTPUT_DIR, 'tasks.db')
CACHE_FILE = os.path.join(OUTPUT_DIR, 'cache.db')
LOG_FILE = os.path.join(OUTPUT_DIR, 'generator.log')

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Database functions
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            text TEXT NOT NULL,
            status TEXT NOT NULL,
            answer TEXT,
            FOREIGN KEY(parent_id) REFERENCES Questions(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            task_type TEXT NOT NULL,
            parameters TEXT,
            status TEXT NOT NULL,
            outcome TEXT,
            FOREIGN KEY(question_id) REFERENCES Questions(id)
        )
    ''')
    conn.commit()
    return conn

def insert_question(conn, text, parent_id=None, status='pending', answer=None):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Questions (text, parent_id, status, answer)
        VALUES (?, ?, ?, ?)
    ''', (text, parent_id, status, answer))
    conn.commit()
    return cursor.lastrowid

def update_question_status(conn, question_id, status, answer=None):
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Questions SET status = ?, answer = ? WHERE id = ?
    ''', (status, answer, question_id))
    conn.commit()

def insert_task(conn, question_id, task_type, parameters, status='pending', outcome=None):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Tasks (question_id, task_type, parameters, status, outcome)
        VALUES (?, ?, ?, ?, ?)
    ''', (question_id, task_type, json.dumps(parameters), status, json.dumps(outcome)))
    conn.commit()
    return cursor.lastrowid

def update_task_status(conn, task_id, status, outcome=None):
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Tasks SET status = ?, outcome = ? WHERE id = ?
    ''', (status, json.dumps(outcome), task_id))
    conn.commit()

def get_existing_task_outcome(conn, task_type, parameters):
    cursor = conn.cursor()
    parameters_json = json.dumps(parameters, sort_keys=True)
    cursor.execute('''
        SELECT outcome FROM Tasks
        WHERE task_type = ? AND parameters = ? AND status = 'completed'
    ''', (task_type, parameters_json))
    row = cursor.fetchone()
    if row:
        return json.loads(row[0]) if row[0] else None
    return None

# Define task and sub-question generation functions with structured output
def generate_tasks_and_subquestions(question_text, cache):
    task_list = tasks.list_tasks()
    logger.debug(f"Generating tasks and subquestions for question: {question_text}")
    prompt = {
        "task_generation": {
            "description": (
                "Please analyze the question below and decompose it into smaller, more specific sub-questions that can help in answering the main question comprehensively. "
                "For each sub-question, consider what tasks might be necessary to answer it, such as performing search queries, extracting content, summarizing information, etc. "
                "Generate a list of tasks with appropriate parameters needed to answer the sub-questions and the main question. "
                "Return ONLY valid JSON with three keys: 'answer' (a string, can be empty), 'tasks' (a list of dictionaries with 'name' and 'parameters'), 'subquestions' (a list of strings). "
                "Ensure that tasks are practical and can be executed by the system. No additional text or explanations."
            ),
            "parameters": {"question": question_text},
            "task_list": task_list,
        }
    }
    prompt_str = json.dumps(prompt)
    response = send_llm_request(prompt_str, cache, MODEL_NAME, OLLAMA_URL)
    # Extract answer, tasks, and subquestions
    answer = response.get("answer", "")
    tasks_data = response.get("tasks", [])
    sub_questions = response.get("subquestions", [])
    return answer, tasks_data, sub_questions

def generate_answer_from_context(context, cache):
    logger.debug("Generating answer from context.")
    prompt = {
        "answer_generation": {
            "description": (
                "Based on the provided context, which includes the question, tasks with their parameters and outcomes, and sub-answers, generate an answer to the question. "
                "Ensure all sources are verified and include concrete URL references. Return ONLY valid JSON with one key: 'answer' (a string). No additional text or explanations."
            ),
            "parameters": context
        }
    }
    prompt_str = json.dumps(prompt)
    response = send_llm_request(prompt_str, cache, MODEL_NAME, OLLAMA_URL)
    answer = response.get('answer', "")
    return answer

# Task Execution
def execute_task(task_name, parameters, cache=None):
    task_function = tasks.TASK_FUNCTIONS.get(task_name)

    # Log task_name and parameters to debug
    logger.debug(f"Executing task: {task_name}, with parameters: {parameters}")
    if not isinstance(parameters, dict):
        logger.error(f"Expected parameters to be a dictionary, got {type(parameters)}")
        raise TypeError(f"Expected parameters to be a dictionary, got {type(parameters)}")

    if task_function:
        return task_function(parameters, cache)
    else:
        logger.error(f"No function found for task: {task_name}")
        raise ValueError(f"No function found for task: {task_name}")

def process_task(task, question_id, conn, cache):
    task_name = task.get("name")
    parameters = task.get("parameters", {})

    logger.debug(f"Processing task '{task_name}' with parameters {parameters}")

    # Check if the task has already been performed with the same parameters
    existing_outcome = get_existing_task_outcome(conn, task_name, parameters)
    if existing_outcome is not None:
        logger.info(f"Using existing outcome for task '{task_name}' with parameters {parameters}")
        outcome = existing_outcome
    else:
        try:
            outcome = execute_task(task_name, parameters, cache)
            # Insert the task with the outcome
            task_id = insert_task(conn, question_id, task_name, parameters, status='completed', outcome=outcome)
            update_task_status(conn, task_id, 'completed', outcome)
        except Exception as e:
            logger.exception(f"Error executing task '{task_name}' with parameters {parameters}: {e}")
            outcome = {'error': str(e)}

    # Now, depending on the task and outcome, generate further tasks
    sub_task_outcomes = []

    if 'error' in outcome:
        logger.error(f"Error in task '{task_name}': {outcome['error']}")
        # Decide how to handle the error
        # For now, include the error in the sub_task_outcomes
        sub_task_outcomes.append({
            'task_name': task_name,
            'parameters': parameters,
            'error': outcome['error']
        })
    else:
        if task_name == 'search_query':
            # For each search result, create 'extract_content' and 'summarize_text' tasks
            search_results = outcome.get('search_results', [])
            for result in search_results:
                url = result.get('url')
                sub_task = {
                    'name': 'extract_content',
                    'parameters': {'url': url}
                }
                sub_outcome = process_task(sub_task, question_id, conn, cache)
                if 'error' in sub_outcome.get('outcome', {}):
                    logger.error(f"Error in sub-task 'extract_content': {sub_outcome['outcome']['error']}")
                    sub_task_outcomes.append({
                        'url': url,
                        'error': sub_outcome['outcome']['error']
                    })
                    continue
                structured_data = sub_outcome.get('outcome', {}).get('structured_data')
                if structured_data:
                    summarize_task = {
                        'name': 'summarize_text',
                        'parameters': {'text': json.dumps(structured_data)}
                    }
                    summary_outcome = process_task(summarize_task, question_id, conn, cache)
                    if 'error' in summary_outcome.get('outcome', {}):
                        logger.error(f"Error in sub-task 'summarize_text': {summary_outcome['outcome']['error']}")
                        sub_task_outcomes.append({
                            'url': url,
                            'error': summary_outcome['outcome']['error']
                        })
                        continue
                    # Collect summaries
                    sub_task_outcomes.append({
                        'url': url,
                        'summary': summary_outcome.get('outcome', {}).get('summary')
                    })
                else:
                    logger.warning(f"No structured data extracted from URL: {url}")
                    sub_task_outcomes.append({
                        'url': url,
                        'error': 'No structured data extracted'
                    })
        elif task_name == 'validate_fact':
             # Check for errors in outcome
            if 'error' in outcome:
                logger.error(f"Error in task '{task_name}': {outcome['error']}")
                sub_task_outcomes.append({
                    "task_name": task_name,
                    "statement": parameters.get("statement"),
                    "error": outcome['error']
                })
            else:
                # If outcome contains validation result, add it to sub_task_outcomes
                is_true = outcome.get("is_true", False)
                confidence = outcome.get("confidence", 0.0)
                sources = outcome.get("sources", [])
                
                sub_task_outcomes.append({
                    "task_name": task_name,
                    "statement": parameters.get("statement"),
                    "is_true": is_true,
                    "confidence": confidence,
                    "sources": sources
                })

    # Return the final outcome, possibly including sub_task_outcomes
    return {
        'task_name': task_name,
        'parameters': parameters,
        'outcome': outcome,
        'sub_tasks': sub_task_outcomes
    }

# Recursive question processing
def process_question(question_id, conn, cache):
    cursor = conn.cursor()
    cursor.execute('SELECT text, status, answer FROM Questions WHERE id = ?', (question_id,))
    question_row = cursor.fetchone()
    if not question_row:
        logger.error(f"Question ID {question_id} not found.")
        return None
    question_text, status, existing_answer = question_row

    logger.debug(f"Processing question ID {question_id}, status: {status}")

    # If the question is already answered, return the answer
    if status == 'answered' and existing_answer:
        logger.info(f"Question ID {question_id} is already answered.")
        return existing_answer

    # Generate tasks, subquestions, and attempt to answer
    answer, tasks_to_perform, sub_questions = generate_tasks_and_subquestions(question_text, cache)

    logger.debug(f"Generated answer: {answer}")
    logger.debug(f"Generated tasks: {tasks_to_perform}")
    logger.debug(f"Generated sub-questions: {sub_questions}")

    # If answer is provided, update question and return
    if answer:
        update_question_status(conn, question_id, 'answered', answer)
        return answer

    # Initialize task outcomes
    task_outcomes = []

    # Process tasks recursively
    for task in tasks_to_perform:
        try:
            outcome = process_task(task, question_id, conn, cache)
            task_outcomes.append(outcome)
        except Exception as e:
            logger.exception(f"Error processing task {task}: {e}")
            task_outcomes.append({
                'task_name': task.get('name'),
                'parameters': task.get('parameters'),
                'error': str(e)
            })

    # Process sub-questions
    sub_answers = []
    for sub_question_text in sub_questions:
        # Check if sub-question already exists to avoid duplicates
        cursor.execute('SELECT id, status, answer FROM Questions WHERE text = ? AND parent_id = ?', (sub_question_text, question_id))
        result = cursor.fetchone()
        if result:
            sub_question_id, sub_status, sub_answer = result
            if sub_status == 'answered' and sub_answer:
                sub_answers.append(sub_answer)
                continue
        else:
            sub_question_id = insert_question(conn, sub_question_text, parent_id=question_id)
        sub_answer = process_question(sub_question_id, conn, cache)
        if sub_answer:
            sub_answers.append(sub_answer)
        else:
            sub_answers.append(f"Could not find an answer to sub-question: {sub_question_text}")

    # After processing tasks and subquestions, attempt to answer the question using the collected information
    combined_context = {
        'question': question_text,
        'tasks': task_outcomes,
        'sub_answers': sub_answers
    }
    answer = generate_answer_from_context(combined_context, cache)
    if answer:
        update_question_status(conn, question_id, 'answered', answer)
        return answer
    else:
        # If still cannot answer, mark as pending
        update_question_status(conn, question_id, 'pending')
        return None

# Main execution flow
if __name__ == '__main__':
    conn = init_db()

    # Open a persistent cache with shelve
    with shelve.open(CACHE_FILE) as cache:
        try:
            question_text = """**Objective**: Create an elaborate complete markdown report detailing advancements in artificial intelligence only in October 2024, covering hardware, software, open-source developments and emerging trends (focus multiple large companies have shown recently) from reputable and credible sources. Ensure the report highlights recent trends and innovations that reflect the latest industry shifts and focus areas. Each statement should include online references to credible sources. Structure the report to appeal to a broad audience, including both technical and strategic stakeholders. Each section should be engaging, visual, and supported by concrete data from authoritative sources, preferably the official announcements, technical documentation, or product pages of the service providers or manufacturers.
                **AI Hardware Advancements**: 
                - Present major updates in AI-specific hardware, focusing on recent breakthroughs and trends:
                    - Summarize upcoming releases or breakthroughs (e.g., new NVIDIA GPUs, Apple’s chips, advancements in edge AI hardware).
                    - Provide performance comparisons to previous models to highlight improvements, efficiency gains, or scalability enhancements.
                    - Include new use cases or efficiency gains expected from these advancements, and discuss how they reflect recent industry trends.

                **Software Innovations**: 
                - Outline cutting-edge software models and updates, including popular trends in AI applications:
                    - Detail improvements in reasoning, multimodal capabilities, and efficiency with models like OpenAI’s GPT, Meta’s Llama, Google’s Gemini, and others that are driving new AI capabilities.
                    - Emphasize recent developments in responsible AI, ethical AI practices, and any alignment improvements in popular models.
                    - Include relevant benchmarks, unique features (such as increased context windows, enhanced image/video processing), and visual comparisons, highlighting recent improvements and trends.

                **Open-Source Contributions**: 
                - Showcase significant open-source AI releases, focusing on recent contributions and trends. Consider for example new open models and AI related frameworks. Consider LLMs and image generation models and other types when applicable.:
                    - Highlight contributions from companies like Meta, Microsoft, Google, and others, explaining the anticipated impact and what is innovative about these tools.
                    - Include real-world applications and potential impact, particularly in underrepresented regions or for solving specific societal challenges, showcasing the relevance to current global AI trends.

                **Validation and Accuracy**: 
                - Ensure all data points are accurate, verified, and from reputable sources. Avoid unverified claims by cross-referencing with multiple credible sources, primarily direct statements from the companies or technical documentation.
            """
            main_question_id = insert_question(conn, question_text)
            main_answer = process_question(main_question_id, conn, cache)
            if main_answer:
                print(f"Answer to the main question: {main_answer}")
            else:
                print("Could not find an answer to the main question.")
        except Exception as e:
            logger.exception("An error occurred during main execution.")
            print(f"An error occurred: {e}")
        finally:
            conn.close()
