# generator.py

import json
import os
import logging
import shelve
import sqlite3
from typing import List, Dict, Any, Optional
import tasks  # Import the tasks module
import database  # Import the new database module
from llm_utils import send_llm_request

# -------------------- Logging Configuration --------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers if they don't already exist
if not logger.handlers:
    # Define the output directory
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define the path for the log file
    LOG_FILE = os.path.join(OUTPUT_DIR, 'generator.log')
    
    # Create a file handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Define logging format
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# -------------------- Constants --------------------

OLLAMA_URL = "http://localhost:11434/api/generate"  # Replace with your actual endpoint
TASK_GENERATION_MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"  # Configurable model name for task generation
ANSWER_GENERATION_MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"  # Configurable model name for answer generation
CACHE_FILE = os.path.join("output", 'cache.db')

# -------------------- Task and Sub-question Generation Functions --------------------

def generate_tasks_and_subquestions(question_text: str, cache: Any) -> (str, List[Dict[str, Any]], List[str]):
    """
    Generate tasks and sub-questions based on the main question.

    Args:
        question_text (str): The main question text.
        cache (Any): The cache object.

    Returns:
        Tuple[str, List[Dict[str, Any]], List[str]]: A tuple containing the answer, list of tasks, and list of sub-questions.
    """
    task_list = tasks.list_tasks()
    logger.debug(f"Generating tasks and subquestions for question: {question_text}")
    prompt = {
        "task_generation": {
            "description": (
                "Please analyze the question below and decompose it into smaller, more specific sub-questions that can help in answering the main question comprehensively. "
                "For each sub-question, consider what tasks might be necessary to answer it, such as performing search queries, extracting content, summarizing information, etc. "
                "Generate a list of tasks with appropriate parameters needed to answer the sub-questions and the main question. "
                "Note that the 'search_query' task can accept an optional 'from_date' parameter (can only be 'past month' or 'past year') to limit search results to recent information. Use this parameter when it is important to get the latest data. "
                "Return ONLY valid JSON with three keys: 'answer' (a string, can be empty), 'tasks' (a list of dictionaries with 'name' and 'parameters'), 'subquestions' (a list of strings). "
                "Ensure that tasks are practical and can be executed by the system. No additional text or explanations."
            ),
            "parameters": {"question": question_text},
            "task_list": task_list,
        }
    }
    prompt_str = json.dumps(prompt)
    response = send_llm_request(prompt_str, cache, TASK_GENERATION_MODEL_NAME, OLLAMA_URL, expect_json=True)
    # Extract answer, tasks, and subquestions
    answer = response.get("answer", "")
    tasks_data = response.get("tasks", [])
    sub_questions = response.get("subquestions", [])
    return answer, tasks_data, sub_questions

def generate_answer_from_context(context: Dict[str, Any], cache: Any) -> str:
    """
    Generate an answer based on the provided context.

    Args:
        context (Dict[str, Any]): The context containing question, tasks, and sub-answers.
        cache (Any): The cache object.

    Returns:
        str: The generated answer.
    """
    logger.debug("Generating answer from context.")
    prompt = {
        "answer_generation": {
            "description": (
                "Based on the provided context, which includes the question, tasks with their parameters and outcomes, and sub-answers, generate an answer to the question. "
                "Ensure all sources are verified and include concrete URL references. "
                "Return ONLY valid JSON with one key: 'answer' (a string). No additional text or explanations."
            ),
            "parameters": context
        }
    }
    prompt_str = json.dumps(prompt)
    response = send_llm_request(prompt_str, cache, ANSWER_GENERATION_MODEL_NAME, OLLAMA_URL, expect_json=True)
    answer = response.get('answer', "")
    return answer

def generate_initial_tasks(question_text: str, keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Generate the initial set of tasks based on the main question and extracted keywords.

    Args:
        question_text (str): The main question text.
        keywords (Optional[List[str]], optional): Extracted keywords from the question. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of initial tasks to perform.
    """
    logger.debug(f"Generating initial tasks for question: {question_text}")
    # The initial task is to perform a search query with the keywords as the query
    if keywords:
        query = ' '.join(keywords)
    else:
        query = question_text
    tasks_to_perform = [
        {
            "name": "search_query",
            "parameters": {"query": query}
        }
    ]
    return tasks_to_perform

def generate_subquestions(question_text: str, cache: Any) -> List[str]:
    """
    Generate sub-questions to further break down the main question.

    Args:
        question_text (str): The main question text.
        cache (Any): The cache object.

    Returns:
        List[str]: A list of sub-questions.
    """
    logger.debug(f"Generating subquestions for question: {question_text}")
    prompt = {
        "subquestion_generation": {
            "description": (
                "Please analyze the question below and decompose it into smaller, more specific sub-questions that can help in answering the main question comprehensively. "
                "Return ONLY valid JSON with one key: 'subquestions' (a list of strings). No additional text or explanations."
            ),
            "parameters": {"question": question_text}
        }
    }
    prompt_str = json.dumps(prompt)
    response = send_llm_request(prompt_str, cache, TASK_GENERATION_MODEL_NAME, OLLAMA_URL, expect_json=True)
    sub_questions = response.get("subquestions", [])
    return sub_questions

# -------------------- Task Execution --------------------

def process_task(task: Dict[str, Any], question_id: int, question_text: str, conn: sqlite3.Connection, cache: Any) -> Dict[str, Any]:
    """
    Process a single task:
    - Check if the task outcome is cached.
    - If not, execute the task using tasks.execute_task.
    - Handle outcomes and potential sub-tasks.

    Args:
        task (Dict[str, Any]): The task dictionary containing 'name' and 'parameters'.
        question_id (int): The ID of the associated question.
        question_text (str): The text of the main question.
        conn (sqlite3.Connection): The database connection object.
        cache (Any): The cache object.

    Returns:
        Dict[str, Any]: The outcome of the task, including any sub-task outcomes.
    """
    task_name = task.get("name")
    parameters = task.get("parameters", {})

    logger.debug(f"Processing task '{task_name}' with parameters {parameters}")

    # Add the question to the parameters if not already present
    parameters.setdefault('question', question_text)

    # Check if the task has already been performed with the same parameters
    existing_outcome = database.get_existing_task_outcome(conn, task_name, parameters)
    if existing_outcome is not None:
        logger.info(f"Using existing outcome for task '{task_name}' with parameters {parameters}")
        outcome = existing_outcome
        # Update the task status to 'completed' if not already done
        cursor = conn.cursor()
        parameters_json = json.dumps(parameters, sort_keys=True)
        cursor.execute('''
            UPDATE Tasks SET status = ?, outcome = ? WHERE question_id = ? AND task_type = ? AND parameters = ?
        ''', ('completed', json.dumps(outcome), question_id, task_name, parameters_json))
        conn.commit()
    else:
        try:
            # Delegate task execution to tasks.py
            outcome = tasks.execute_task(task_name, parameters, cache)
            # Insert the task with the outcome
            task_id = database.insert_task(conn, question_id, task_name, parameters, status='completed', outcome=outcome)
            database.update_task_status(conn, task_id, 'completed', outcome)
        except Exception as e:
            logger.exception(f"Error executing task '{task_name}' with parameters {parameters}: {e}")
            outcome = {'error': str(e)}
            # Insert the task with the error
            task_id = database.insert_task(conn, question_id, task_name, parameters, status='failed', outcome=outcome)
            database.update_task_status(conn, task_id, 'failed', outcome)

    # Now, depending on the task and outcome, generate further tasks
    sub_task_outcomes = []

    if 'error' in outcome:
        logger.error(f"Error in task '{task_name}': {outcome['error']}")
        # Decide how to handle the error
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
                    'parameters': {'url': url, 'question': question_text}
                }
                sub_outcome = process_task(sub_task, question_id, question_text, conn, cache)
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
                        'parameters': {'text': structured_data, 'question': question_text}
                    }
                    summary_outcome = process_task(summarize_task, question_id, question_text, conn, cache)
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
            if 'error' in outcome:
                logger.error(f"Error in task '{task_name}': {outcome['error']}")
                sub_task_outcomes.append({
                    "task_name": task_name,
                    "statement": parameters.get("statement"),
                    "error": outcome['error']
                })
            else:
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

# -------------------- Recursive Question Processing --------------------

def process_question(question_id: int, conn: sqlite3.Connection, cache: Any, attempts: int = 0, max_attempts: int = 3) -> Optional[str]:
    """
    Recursively process a question by executing tasks and handling sub-questions.

    Args:
        question_id (int): The ID of the question to process.
        conn (sqlite3.Connection): The database connection object.
        cache (Any): The cache object.
        attempts (int, optional): Current attempt count. Defaults to 0.
        max_attempts (int, optional): Maximum allowed attempts. Defaults to 3.

    Returns:
        Optional[str]: The answer to the question if found, else None.
    """
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

    # If we have exceeded max attempts, stop processing
    if attempts >= max_attempts:
        logger.warning(f"Max attempts reached for question ID {question_id}.")
        database.update_question_status(conn, question_id, 'unanswerable')
        return None

    # Initialize task outcomes and sub_answers
    task_outcomes = []
    sub_answers = []

    # Step 1: Extract keywords from the question
    keyword_task = {
        "name": "extract_keywords",
        "parameters": {"text": question_text}
    }
    keyword_outcome = process_task(keyword_task, question_id, question_text, conn, cache)
    if 'error' in keyword_outcome.get('outcome', {}):
        logger.error(f"Error extracting keywords: {keyword_outcome['outcome']['error']}")
        database.update_question_status(conn, question_id, 'unanswerable')
        return None
    keywords = keyword_outcome.get('outcome', {}).get('keywords', [])
    logger.debug(f"Extracted keywords: {keywords}")

    # Step 2: Evaluate if the question is focused
    evaluate_task = {
        "name": "evaluate_question_focus",
        "parameters": {"question": question_text}
    }
    evaluate_outcome = process_task(evaluate_task, question_id, question_text, conn, cache)
    if 'error' in evaluate_outcome.get('outcome', {}):
        logger.error(f"Error evaluating question focus: {evaluate_outcome['outcome']['error']}")
        database.update_question_status(conn, question_id, 'unanswerable')
        return None
    is_focused = evaluate_outcome.get('outcome', {}).get('is_focused', False)
    logger.debug(f"Is the question focused? {is_focused}")

    if is_focused:
        # Proceed with generating initial tasks based on keywords
        tasks_to_perform = generate_initial_tasks(question_text, keywords=keywords)
        logger.debug(f"Generated initial tasks based on keywords: {tasks_to_perform}")

        # Process tasks recursively
        for task in tasks_to_perform:
            try:
                outcome = process_task(task, question_id, question_text, conn, cache)
                task_outcomes.append(outcome)
            except Exception as e:
                logger.exception(f"Error processing task {task}: {e}")
                task_outcomes.append({
                    'task_name': task.get('name'),
                    'parameters': task.get('parameters'),
                    'error': str(e)
                })

        # Attempt to generate an answer from the context
        combined_context = {
            'question': question_text,
            'tasks': task_outcomes,
            'sub_answers': sub_answers  # Empty at this point
        }
        answer = generate_answer_from_context(combined_context, cache)
        if answer:
            database.update_question_status(conn, question_id, 'answered', answer)
            return answer

        # If unable to generate answer, generate subquestions
        logger.debug("Unable to generate answer from initial tasks, generating subquestions.")

    else:
        # If question is not focused, generate subquestions first
        logger.debug("Question is not focused. Generating subquestions.")

    # Generate subquestions regardless of focus
    sub_questions = generate_subquestions(question_text, cache)
    logger.debug(f"Generated sub-questions: {sub_questions}")

    # Process sub-questions
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
            sub_question_id = database.insert_question(conn, sub_question_text, parent_id=question_id)
        sub_answer = process_question(sub_question_id, conn, cache, attempts=attempts+1, max_attempts=max_attempts)
        if sub_answer:
            sub_answers.append(sub_answer)
        else:
            sub_answers.append(f"Could not find an answer to sub-question: {sub_question_text}")

    # After processing subquestions, attempt to generate an answer using the collected information
    combined_context = {
        'question': question_text,
        'tasks': task_outcomes,
        'sub_answers': sub_answers
    }
    answer = generate_answer_from_context(combined_context, cache)
    if answer:
        database.update_question_status(conn, question_id, 'answered', answer)
        return answer
    else:
        # Check if there are any pending tasks or subquestions
        cursor.execute('SELECT COUNT(*) FROM Tasks WHERE question_id = ? AND status = ?', (question_id, 'pending'))
        pending_tasks_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM Questions WHERE parent_id = ? AND status != ?', (question_id, 'answered'))
        pending_subquestions_count = cursor.fetchone()[0]

        if pending_tasks_count == 0 and pending_subquestions_count == 0:
            logger.debug("No more pending tasks or subquestions, attempting to generate more.")
            # Recurse with increased attempts
            return process_question(question_id, conn, cache, attempts=attempts+1, max_attempts=max_attempts)
        else:
            logger.debug("There are still pending tasks or subquestions.")
            # Return None to indicate that processing is ongoing
            return None

# -------------------- Main Execution Flow --------------------

if __name__ == '__main__':
    # Initialize the database
    conn = database.init_db()

    # Open a persistent cache with shelve
    with shelve.open(CACHE_FILE) as cache:
        try:
            # Define the main question
            question_text = "what is the meaning of life"
            logger.info(f"Main question: {question_text}")

            # Insert the main question into the database
            main_question_id = database.insert_question(conn, question_text)

            # Process the main question
            main_answer = process_question(main_question_id, conn, cache)

            if main_answer:
                print(f"Answer to the main question: {main_answer}")
                logger.info("Successfully obtained an answer to the main question.")
            else:
                print("Could not find an answer to the main question.")
                logger.warning("Failed to obtain an answer to the main question.")
        except Exception as e:
            logger.exception("An error occurred during main execution.")
            print(f"An error occurred: {e}")
        finally:
            conn.close()
            logger.debug("Database connection closed.")
