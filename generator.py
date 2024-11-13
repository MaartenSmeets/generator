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
ANSWER_GENERATION_MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"  # Configurable model name for answer generation
GENERATE_SUBQUESTIONS_MODEL_NAME = "llama3.1:70b-instruct-q4_K_M"  # Configurable model name for subquestion generation
CACHE_FILE = os.path.join("output", 'cache.db')
MAX_ATTEMPTS = 3  # Increased from 2 to 3
MAX_DEPTH = 5

# -------------------- Helper Functions --------------------

def load_cache(cache_file: str):
    return shelve.open(cache_file)

# -------------------- Classes --------------------

class TaskManager:
    def __init__(self, conn: sqlite3.Connection, cache: shelve.Shelf):
        self.conn = conn
        self.cache = cache

    def process_task(self, task: Dict[str, Any], question_id: Optional[int], question_text: str) -> Dict[str, Any]:
        task_name = task.get("name")
        parameters = task.get("parameters", {})
        logger.debug(f"Processing task '{task_name}' with parameters {parameters}")

        # Ensure parameters are JSON-serializable
        parameters = json.loads(json.dumps(parameters))

        # Add the question to the parameters if not already present
        parameters.setdefault('question', question_text)

        # Check if the task has already been performed with the same parameters
        existing_outcome = database.get_existing_task_outcome(self.conn, task_name, parameters)
        if existing_outcome is not None:
            logger.info(f"Using existing outcome for task '{task_name}' with parameters {parameters}")
            outcome = existing_outcome
            # Update the task status to 'completed' if not already done
            cursor = self.conn.cursor()
            parameters_json = json.dumps(parameters, sort_keys=True)
            cursor.execute('''
                UPDATE Tasks SET status = ?, outcome = ? WHERE question_id = ? AND task_type = ? AND parameters = ?
            ''', ('completed', json.dumps(outcome), question_id, task_name, parameters_json))
            self.conn.commit()
        else:
            try:
                # Insert the task as pending
                task_id = database.insert_task(self.conn, question_id, task_name, parameters, status='pending', outcome={})
                # Delegate task execution to tasks.py
                outcome = tasks.execute_task(task_name, parameters, self.cache)
                # Update the task with the outcome
                database.update_task_status(self.conn, task_id, 'completed', outcome)
            except Exception as e:
                logger.exception(f"Error executing task '{task_name}' with parameters {parameters}: {e}")
                outcome = {'error': str(e)}
                # Update the task with the error
                database.update_task_status(self.conn, task_id, 'failed', outcome)

        # Handle task-specific logic
        return self.handle_task_outcome(task_name, outcome, parameters, question_id, question_text)

    def handle_task_outcome(self, task_name: str, outcome: Dict[str, Any], parameters: Dict[str, Any], question_id: Optional[int], question_text: str) -> Dict[str, Any]:
        sub_task_outcomes = []

        if 'error' in outcome:
            logger.error(f"Error in task '{task_name}': {outcome['error']}")
            sub_task_outcomes.append({
                'task_name': task_name,
                'parameters': parameters,
                'error': outcome['error']
            })
            return {
                'task_name': task_name,
                'parameters': parameters,
                'outcome': outcome,
                'sub_tasks': sub_task_outcomes
            }

        if task_name == 'search_query':
            search_results = outcome.get('search_results', [])
            for result in search_results:
                url = result.get('url')
                if not url:
                    logger.warning("Search result missing 'url' field.")
                    continue
                # Extract Content Task
                extract_task = {
                    'name': 'extract_content',
                    'parameters': {'url': url, 'question': question_text}
                }
                extract_outcome = self.process_task(extract_task, question_id, question_text)
                if 'error' in extract_outcome.get('outcome', {}):
                    logger.error(f"Error in 'extract_content' task: {extract_outcome['outcome']['error']}")
                    sub_task_outcomes.append({
                        'url': url,
                        'error': extract_outcome['outcome']['error']
                    })
                    continue
                extracted_text = extract_outcome.get('outcome', {}).get('extracted_text')
                if extracted_text:
                    # Summarize Text Task
                    summarize_task = {
                        'name': 'summarize_text',
                        'parameters': {'text': extracted_text, 'question': question_text}
                    }
                    summary_outcome = self.process_task(summarize_task, question_id, question_text)
                    if 'error' in summary_outcome.get('outcome', {}):
                        logger.error(f"Error in 'summarize_text' task: {summary_outcome['outcome']['error']}")
                        sub_task_outcomes.append({
                            'url': url,
                            'error': summary_outcome['outcome']['error']
                        })
                        continue
                    summary = summary_outcome.get('outcome', {}).get('summary')
                    # Evaluate Summary Task
                    evaluate_task = {
                        'name': 'evaluate_summary',
                        'parameters': {'summary': summary, 'question': question_text}
                    }
                    evaluation_outcome = self.process_task(evaluate_task, question_id, question_text)
                    if 'error' in evaluation_outcome.get('outcome', {}):
                        logger.error(f"Error in 'evaluate_summary' task: {evaluation_outcome['outcome']['error']}")
                        is_valuable = True  # Default to valuable if unsure
                    else:
                        is_valuable = evaluation_outcome.get('outcome', {}).get('is_valuable', True)
                        if is_valuable is None:
                            is_valuable = True
                    if is_valuable:
                        sub_task_outcomes.append({
                            'url': url,
                            'summary': summary,
                            'valuable': True
                        })
                    else:
                        logger.debug(f"Summary deemed not valuable for URL: {url}")
                        sub_task_outcomes.append({
                            'url': url,
                            'summary': summary,
                            'valuable': False
                        })
                else:
                    logger.warning(f"No extracted text from URL: {url}")
                    sub_task_outcomes.append({
                        'url': url,
                        'error': 'No extracted text.'
                    })

        elif task_name in ['validate_fact', 'is_reputable_source', 'evaluate_summary']:
            # Handle other task-specific outcomes if necessary
            sub_task_outcomes.append({
                'task_name': task_name,
                'outcome': outcome
            })

        # Add more task-specific handling as needed

        return {
            'task_name': task_name,
            'parameters': parameters,
            'outcome': outcome,
            'sub_tasks': sub_task_outcomes
        }

class AnswerGenerator:
    def __init__(self, cache: shelve.Shelf):
        self.cache = cache

    def generate_answer_from_context(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug("Generating answer from context.")

        # Build the context string from valuable summaries and sub-answers
        context_strings = []

        valuable_summaries = context.get('valuable_summaries', [])
        if valuable_summaries:
            context_strings.append("Valuable Summaries:")
            for summary in valuable_summaries:
                context_strings.append(f"- {summary}")

        sub_answers = context.get('sub_answers', [])
        if sub_answers:
            context_strings.append("\nSub-Answers:")
            for sub in sub_answers:
                context_strings.append(f"Question: {sub['question']}\nAnswer: {sub['answer']}")

        context_str = '\n'.join(context_strings)

        prompt = (
            "Using ONLY the provided context, generate a comprehensive and detailed answer to the question. "
            "Do NOT use any prior knowledge or data not included in the context. "
            "If the context contains enough information for a partial but incomplete answer, provide the partial answer in the 'answer' field and explain what additional information is missing in the 'missing_information' field. "
            "If the context does not contain sufficient information to answer the question at all, set 'answer' to an empty string and describe in detail what specific information is missing in 'missing_information'. "
            "Return ONLY valid JSON with two keys: 'answer' (a string) and 'missing_information' (a string explaining what is missing). "
            "If the question is fully answerable, set 'missing_information' to an empty string. "
            "Do not include any additional text or explanations.\n\n"
            "Examples:\n"
            "{\n"
            '  "answer": "Your answer here.",\n'
            '  "missing_information": ""\n'
            "}\n"
            "{\n"
            '  "answer": "Partial answer here.",\n'
            '  "missing_information": "The context lacks specific details about the implementation steps."\n'
            "}\n"
            "{\n"
            '  "answer": "",\n'
            '  "missing_information": "The context lacks details about the latest AI hardware advancements."\n'
            "}\n\n"
            f"Question:\n{question}\n\nContext:\n{context_str}\n\nAnswer:"
        )

        try:
            response = send_llm_request(
                prompt,
                self.cache,
                ANSWER_GENERATION_MODEL_NAME,
                OLLAMA_URL,
                expect_json=True  # Expecting a JSON response
            )
            logger.debug(f"LLM response for answer generation: {response}")

            # Validate and extract the 'answer' and 'missing_information' fields from the JSON response
            answer = response.get("answer", "")
            missing_information = response.get("missing_information", "")
            if not isinstance(answer, str) or not isinstance(missing_information, str):
                logger.error("The 'answer' or 'missing_information' field is missing or not a string in the JSON response.")
                return {}
            logger.debug(f"Generated answer: {answer}")
            logger.debug(f"Missing information: {missing_information}")
            return {"answer": answer, "missing_information": missing_information}

        except Exception as e:
            logger.exception(f"An unexpected error occurred while generating answer from context: {e}")
            return {}

class QuestionProcessor:
    def __init__(self, conn: sqlite3.Connection, cache: shelve.Shelf):
        self.conn = conn
        self.cache = cache
        self.task_manager = TaskManager(conn, cache)
        self.answer_generator = AnswerGenerator(cache)

    def get_question_depth(self, question_id: int) -> int:
        """
        Helper method to determine the depth of a question in the hierarchy.
        The main question has a depth of 0.
        """
        depth = 0
        current_id = question_id
        while True:
            cursor = self.conn.cursor()
            cursor.execute('SELECT parent_id FROM Questions WHERE id = ?', (current_id,))
            row = cursor.fetchone()
            if not row or not row[0]:
                break
            depth += 1
            current_id = row[0]
        return depth

    def generate_subquestions_from_missing_information(self, question_id: int, missing_information: str) -> List[str]:
        """
        Generate subquestions based on the missing information.

        Args:
            question_id (int): The ID of the question.
            missing_information (str): The information that is missing to answer the question.

        Returns:
            List[str]: A list of generated subquestions.
        """
        logger.debug(f"Generating subquestions from missing information: {missing_information}")

        # Fetch the main question text from the database
        cursor = self.conn.cursor()
        cursor.execute('SELECT text FROM Questions WHERE id = ?', (question_id,))
        question_row = cursor.fetchone()
        if not question_row:
            logger.error(f"Question ID {question_id} not found.")
            return []
        question_text = question_row[0]

        # Construct prompt for LLM enforcing JSON response
        description = (
            "Based on the following missing information and the main question, generate specific sub-questions that, when answered, "
            "will provide the necessary information to answer the main question."
            "Ensure that the combination of answers to these sub-questions will fully address the missing information."
        )
        expected_keys = {"subquestions": ["List of subquestions as strings"]}
        prompt = tasks.enforce_json_response(description, expected_keys) + f"\n\nMain Question:\n{question_text}\n\nMissing Information:\n{missing_information}"

        try:
            response = send_llm_request(
                prompt,
                self.cache,
                GENERATE_SUBQUESTIONS_MODEL_NAME,
                OLLAMA_URL,
                expect_json=True  # Expecting a JSON response
            )
            if response is None:
                logger.error("Failed to get response from LLM when generating subquestions from missing information.")
                return []

            sub_questions = response.get("subquestions", [])
            if not isinstance(sub_questions, list) or not all(isinstance(q, str) for q in sub_questions):
                logger.error("Subquestions are not in the expected format when generated from missing information.")
                return []
            logger.debug(f"Generated sub-questions from missing information: {sub_questions}")
            return sub_questions
        except Exception as e:
            logger.exception(f"Error generating subquestions from missing information: {e}")
            return []

    def process_question(self, question_id: int, attempts: int = 0) -> Optional[str]:
        cursor = self.conn.cursor()
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
        if attempts >= MAX_ATTEMPTS:
            logger.warning(f"Max attempts reached for question ID {question_id}.")
            database.update_question_status(self.conn, question_id, 'unanswerable')
            return None

        # Determine the current depth of the question
        current_depth = self.get_question_depth(question_id)
        logger.debug(f"Current question depth: {current_depth}")

        # If the maximum depth is reached, do not generate subquestions
        if current_depth >= MAX_DEPTH:
            logger.debug(f"Maximum question depth {MAX_DEPTH} reached. Attempting to generate an answer directly.")
            combined_context = self.collect_context(question_id)
            answer_result = self.answer_generator.generate_answer_from_context(question_text, combined_context)
            if answer_result:
                answer_text = answer_result.get('answer', '')
                missing_information = answer_result.get('missing_information', '')
                # Validate if the answer indicates unanswerable
                is_unanswerable = self.validate_answer(question_id, question_text, answer_text)
                if is_unanswerable is None:
                    logger.error("Failed to validate answer. Proceeding with the available answer.")
                    database.update_question_status(self.conn, question_id, 'answered', answer_text)
                    return answer_text
                elif is_unanswerable:
                    logger.info("Answer indicates the question is unanswerable.")
                    database.update_question_status(self.conn, question_id, 'unanswerable')
                    return None
                else:
                    database.update_question_status(self.conn, question_id, 'answered', answer_text)
                    return answer_text

            logger.debug("Unable to generate answer at maximum depth.")
            database.update_question_status(self.conn, question_id, 'unanswerable')
            return None

        # Step 1: Extract keywords from the question if not already done
        if not database.has_task(self.conn, question_id, 'extract_keywords'):
            keywords = self.extract_keywords(question_id, question_text)
            if keywords is None:
                logger.error("Failed to extract keywords.")
                database.update_question_status(self.conn, question_id, 'unanswerable')
                return None
            logger.debug(f"Extracted keywords: {keywords}")
        else:
            # Fetch the outcome from the existing task
            keywords = database.get_task_outcome(self.conn, question_id, 'extract_keywords').get('keywords', [])
            logger.debug(f"Retrieved existing keywords: {keywords}")

        # Step 2: Evaluate if the question is focused
        if not database.has_task(self.conn, question_id, 'evaluate_question_focus'):
            is_focused = self.evaluate_question_focus(question_id, question_text)
        else:
            # Fetch the outcome from the existing task
            is_focused = database.get_task_outcome(self.conn, question_id, 'evaluate_question_focus').get('is_focused', False)
        logger.debug(f"Is the question focused? {is_focused}")

        task_outcomes = []
        sub_answers = []

        if is_focused:
            # Generate initial tasks based on keywords
            tasks_to_perform = self.generate_initial_tasks(question_text, keywords)
            logger.debug(f"Generated initial tasks based on keywords: {tasks_to_perform}")

            # Process each task
            for task in tasks_to_perform:
                outcome = self.task_manager.process_task(task, question_id, question_text)
                task_outcomes.append(outcome)

            # Check if any summaries are valuable
            any_valuable = any(
                any(sub_task.get('valuable') for sub_task in outcome.get('sub_tasks', []))
                for outcome in task_outcomes
            )
            logger.debug(f"Any valuable summaries found? {any_valuable}")

            if not any_valuable:
                # Treat question as not focused
                logger.info("No valuable summaries found, reconsidering question focus.")
                is_focused = False  # Update the flag to treat as not focused

        if not is_focused:
            logger.debug("Question is not focused. Generating subquestions.")

            # Generate subquestions if not already done
            if not database.has_task(self.conn, question_id, 'generate_subquestions'):
                sub_questions = self.generate_subquestions(question_id, question_text)
                logger.debug(f"Generated sub-questions: {sub_questions}")
                # Save subquestions in the database
                for sub_question_text in sub_questions:
                    self.get_or_create_subquestion(sub_question_text, question_id)
            else:
                # Retrieve existing subquestions from the database
                sub_questions = database.get_subquestions(self.conn, question_id)
                logger.debug(f"Retrieved existing sub-questions: {sub_questions}")

            # Process sub-questions
            cursor.execute('SELECT id FROM Questions WHERE parent_id = ?', (question_id,))
            sub_question_ids = [row[0] for row in cursor.fetchall()]
            for sub_question_id in sub_question_ids:
                sub_answer = self.process_question(sub_question_id, attempts=attempts + 1)
                if sub_answer:
                    sub_answers.append(sub_answer)
                else:
                    # Retrieve sub-question text for logging
                    cursor.execute('SELECT text FROM Questions WHERE id = ?', (sub_question_id,))
                    sub_question_text = cursor.fetchone()[0]
                    sub_answers.append(f"Could not find an answer to sub-question: {sub_question_text}")

        else:
            # Attempt to generate an answer from the context of focused tasks
            combined_context = self.collect_context(question_id)
            answer_result = self.answer_generator.generate_answer_from_context(question_text, combined_context)
            # After attempting to generate an answer
            if answer_result:
                answer_text = answer_result.get('answer', '')
                missing_information = answer_result.get('missing_information', '')

                if not answer_text and missing_information:
                    # Generate subquestions based on missing information
                    logger.info(f"Missing information identified: {missing_information}")
                    # Use missing_information to generate subquestions
                    sub_questions = self.generate_subquestions_from_missing_information(question_id, missing_information)
                    # Save subquestions in the database
                    for sub_question_text in sub_questions:
                        self.get_or_create_subquestion(sub_question_text, question_id)
                    # Re-process the question after adding subquestions
                    return self.process_question(question_id, attempts=attempts + 1)
                elif not answer_text and not missing_information:
                    # Handle the case where both answer and missing_information are empty
                    logger.info("No answer or missing information provided. Marking question as unanswerable.")
                    database.update_question_status(self.conn, question_id, 'unanswerable')
                    return None
                else:
                    # Validate if the answer indicates unanswerable
                    is_unanswerable = self.validate_answer(question_id, question_text, answer_text)
                    if is_unanswerable is None:
                        logger.error("Failed to validate answer. Proceeding with the available answer.")
                        database.update_question_status(self.conn, question_id, 'answered', answer_text)
                        return answer_text
                    elif is_unanswerable:
                        logger.info("Answer indicates the question is unanswerable.")
                        database.update_question_status(self.conn, question_id, 'unanswerable')
                        return None
                    else:
                        database.update_question_status(self.conn, question_id, 'answered', answer_text)
                        return answer_text

            logger.debug("Unable to generate answer from initial tasks, generating subquestions.")
            # Generate subquestions if not already done
            if not database.has_task(self.conn, question_id, 'generate_subquestions'):
                sub_questions = self.generate_subquestions(question_id, question_text)
                logger.debug(f"Generated sub-questions: {sub_questions}")
                # Save subquestions in the database
                for sub_question_text in sub_questions:
                    self.get_or_create_subquestion(sub_question_text, question_id)
            else:
                # Retrieve existing subquestions from the database
                sub_questions = database.get_subquestions(self.conn, question_id)
                logger.debug(f"Retrieved existing sub-questions: {sub_questions}")

            # Process sub-questions
            cursor.execute('SELECT id FROM Questions WHERE parent_id = ?', (question_id,))
            sub_question_ids = [row[0] for row in cursor.fetchall()]
            for sub_question_id in sub_question_ids:
                sub_answer = self.process_question(sub_question_id, attempts=attempts + 1)
                if sub_answer:
                    sub_answers.append(sub_answer)
                else:
                    # Retrieve sub-question text for logging
                    cursor.execute('SELECT text FROM Questions WHERE id = ?', (sub_question_id,))
                    sub_question_text = cursor.fetchone()[0]
                    sub_answers.append(f"Could not find an answer to sub-question: {sub_question_text}")

                # After processing tasks and/or subquestions, attempt to generate an answer using the collected information
                combined_context = self.collect_context(question_id)
                answer_result = self.answer_generator.generate_answer_from_context(question_text, combined_context)

                if answer_result:
                    answer_text = answer_result.get('answer', '')
                    missing_information = answer_result.get('missing_information', '')
                    if missing_information:
                        # Generate subquestions based on missing information
                        logger.info(f"Missing information identified: {missing_information}")
                        # Use missing_information to generate subquestions
                        sub_questions = self.generate_subquestions_from_missing_information(question_id, missing_information)
                        # Save subquestions in the database
                        for sub_question_text in sub_questions:
                            self.get_or_create_subquestion(sub_question_text, question_id)
                        # Re-process the question after adding subquestions
                        return self.process_question(question_id, attempts=attempts + 1)
                    elif answer_text:
                        # Validate if the answer indicates unanswerable
                        is_unanswerable = self.validate_answer(question_id, question_text, answer_text)
                        if is_unanswerable is None:
                            logger.error("Failed to validate answer. Proceeding with the available answer.")
                            database.update_question_status(self.conn, question_id, 'answered', answer_text)
                            return answer_text
                        elif is_unanswerable:
                            logger.info("Answer indicates the question is unanswerable.")
                            database.update_question_status(self.conn, question_id, 'unanswerable')
                            return None
                        else:
                            database.update_question_status(self.conn, question_id, 'answered', answer_text)
                            return answer_text
                    else:
                        # Handle the case where both answer and missing_information are empty
                        logger.info("No answer or missing information provided. Marking question as unanswerable.")
                        database.update_question_status(self.conn, question_id, 'unanswerable')
                        return None
            else:
                # Handle the case where answer generation failed
                # Check for pending tasks or subquestions
                if not self.has_pending_tasks_or_subquestions(question_id):
                    if attempts < MAX_ATTEMPTS:
                        logger.debug("No more pending tasks or subquestions, retrying.")
                        return self.process_question(question_id, attempts=attempts + 1)
                    else:
                        logger.warning(f"Max attempts reached for question ID {question_id}. Storing partial answer if any.")
                        # Store any partial answers collected
                        partial_answer = self.collect_partial_answers(question_id)
                        if partial_answer:
                            database.update_question_status(self.conn, question_id, 'answered', partial_answer)
                            return partial_answer
                        else:
                            database.update_question_status(self.conn, question_id, 'unanswerable')
                            return None
                else:
                    logger.debug("There are still pending tasks or subquestions.")
                    return None

    def validate_answer(self, question_id: int, question: str, answer: str) -> Optional[bool]:
        """
        Validates whether the generated answer indicates that the question is unanswerable.

        Args:
            question_id (int): The ID of the question.
            question (str): The original question text.
            answer (str): The generated answer text.

        Returns:
            Optional[bool]: True if the answer indicates unanswerable, False if answerable, None if validation failed.
        """
        validation_task = {
            "name": "validate_answer",
            "parameters": {"question": question, "answer": answer}
        }
        outcome = self.task_manager.process_task(validation_task, question_id=question_id, question_text=question)
        if 'error' in outcome.get('outcome', {}):
            logger.error(f"Error validating answer: {outcome['outcome']['error']}")
            return None
        is_unanswerable = outcome.get('outcome', {}).get('is_unanswerable', False)
        return is_unanswerable

    def extract_keywords(self, question_id: int, question_text: str) -> Optional[List[str]]:
        keyword_task = {
            "name": "extract_keywords",
            "parameters": {"text": question_text}
        }
        outcome = self.task_manager.process_task(keyword_task, question_id=question_id, question_text=question_text)
        if 'error' in outcome.get('outcome', {}):
            logger.error(f"Error extracting keywords: {outcome['outcome']['error']}")
            return None
        keywords = outcome.get('outcome', {}).get('keywords', [])
        return keywords

    def evaluate_question_focus(self, question_id: int, question_text: str) -> bool:
        evaluate_task = {
            "name": "evaluate_question_focus",
            "parameters": {"question": question_text}
        }
        outcome = self.task_manager.process_task(evaluate_task, question_id=question_id, question_text=question_text)
        if 'error' in outcome.get('outcome', {}):
            logger.error(f"Error evaluating question focus: {outcome['outcome']['error']}")
            return False
        is_focused = outcome.get('outcome', {}).get('is_focused', False)
        return is_focused

    def generate_initial_tasks(self, question_text: str, keywords: List[str]) -> List[Dict[str, Any]]:
        logger.debug(f"Generating initial tasks for question: {question_text}")
        query = ' '.join(keywords) if keywords else question_text
        return [
            {
                "name": "search_query",
                "parameters": {"query": query}
            }
        ]

    def generate_subquestions(self, question_id: int, question_text: str) -> List[str]:
        subquestions_task = {
            "name": "generate_subquestions",
            "parameters": {"question": question_text}
        }
        outcome = self.task_manager.process_task(subquestions_task, question_id=question_id, question_text=question_text)
        if 'error' in outcome.get('outcome', {}):
            logger.error(f"Error generating subquestions: {outcome['outcome']['error']}")
            return []
        sub_questions = outcome.get('outcome', {}).get('subquestions', [])
        return sub_questions

    def get_or_create_subquestion(self, sub_question_text: str, parent_id: int) -> int:
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, status, answer FROM Questions WHERE text = ? AND parent_id = ?', (sub_question_text, parent_id))
        result = cursor.fetchone()
        if result:
            sub_question_id, sub_status, sub_answer = result
            if sub_status == 'answered' and sub_answer:
                logger.info(f"Sub-question already answered: {sub_question_text}")
                return sub_question_id
            else:
                return sub_question_id
        else:
            sub_question_id = database.insert_question(self.conn, sub_question_text, parent_id=parent_id)
            logger.debug(f"Inserted new sub-question with ID {sub_question_id}: {sub_question_text}")
            return sub_question_id

    def has_pending_tasks_or_subquestions(self, question_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM Tasks WHERE question_id = ? AND status = ?', (question_id, 'pending'))
        pending_tasks_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM Questions WHERE parent_id = ? AND status != ?', (question_id, 'answered'))
        pending_subquestions_count = cursor.fetchone()[0]

        return pending_tasks_count > 0 or pending_subquestions_count > 0

    def collect_valuable_summaries(self, question_id: int) -> List[str]:
        """
        Recursively collects valuable summaries for a question and its subquestions.

        Args:
            question_id (int): The ID of the question.

        Returns:
            List[str]: A list of valuable summaries as strings.
        """
        cursor = self.conn.cursor()

        # Collect valuable summaries for the question
        cursor.execute('''
            SELECT S.outcome
            FROM Tasks S
            JOIN Tasks E ON S.question_id = E.question_id AND S.parameters = E.parameters
            WHERE S.task_type = 'summarize_text'
            AND E.task_type = 'evaluate_summary'
            AND S.question_id = ?
            AND E.status = 'completed'
            AND json_extract(E.outcome, '$.is_valuable') = 1
        ''', (question_id,))
        valuable_summaries = []
        for row in cursor.fetchall():
            outcome = json.loads(row[0])
            summary = outcome.get('summary', '')
            if summary:
                valuable_summaries.append(summary)

        # Recursively collect valuable summaries from subquestions
        cursor.execute('SELECT id FROM Questions WHERE parent_id = ?', (question_id,))
        sub_question_ids = [row[0] for row in cursor.fetchall()]
        for sub_question_id in sub_question_ids:
            valuable_summaries.extend(self.collect_valuable_summaries(sub_question_id))

        return valuable_summaries

    def collect_sub_answers(self, question_id: int) -> List[Dict[str, Any]]:
        """
        Recursively collects sub-answers from subquestions.

        Args:
            question_id (int): The ID of the question.

        Returns:
            List[Dict[str, Any]]: A list of sub-answers.
        """
        cursor = self.conn.cursor()
        sub_answers = []

        cursor.execute('SELECT id, text FROM Questions WHERE parent_id = ?', (question_id,))
        sub_questions = cursor.fetchall()
        for sub_id, sub_question_text in sub_questions:
            cursor.execute('SELECT status, answer FROM Questions WHERE id = ?', (sub_id,))
            status, answer = cursor.fetchone()
            if status == 'answered' and answer:
                sub_answers.append({'question': sub_question_text, 'answer': answer})
            # Recursively collect sub_answers from deeper levels
            sub_answers.extend(self.collect_sub_answers(sub_id))

        return sub_answers

    def collect_context(self, question_id: int) -> Dict[str, Any]:
        """
        Collects context from valuable summaries and sub-answers related to the question.

        Args:
            question_id (int): The ID of the question.

        Returns:
            Dict[str, Any]: A dictionary containing the context.
        """
        valuable_summaries = self.collect_valuable_summaries(question_id)
        sub_answers = self.collect_sub_answers(question_id)
        context = {
            'valuable_summaries': valuable_summaries,
            'sub_answers': sub_answers
        }
        return context

    def collect_partial_answers(self, question_id: int) -> Optional[str]:
        """
        Collects partial answers from subquestions.

        Args:
            question_id (int): The ID of the question.

        Returns:
            Optional[str]: A combined partial answer if available.
        """
        cursor = self.conn.cursor()

        # Get sub-answers from subquestions
        cursor.execute('SELECT id FROM Questions WHERE parent_id = ?', (question_id,))
        sub_question_ids = [row[0] for row in cursor.fetchall()]
        sub_answers = []
        for sub_id in sub_question_ids:
            cursor.execute('SELECT status, answer FROM Questions WHERE id = ?', (sub_id,))
            status, answer = cursor.fetchone()
            if status == 'answered' and answer:
                sub_answers.append(answer)

        if sub_answers:
            # Combine sub-answers into a partial answer
            partial_answer = '\n\n'.join(sub_answers)
            return partial_answer
        else:
            return None

# -------------------- Main Execution Flow --------------------

def main():
    # Initialize the database
    conn = database.init_db()

    # Open a persistent cache with shelve
    with load_cache(CACHE_FILE) as cache:
        try:
            # Initialize the question processor
            processor = QuestionProcessor(conn, cache)

            # Check if there are any pending questions
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM Questions WHERE status != 'answered' AND status != 'unanswerable'")
            pending_questions = cursor.fetchall()

            if pending_questions:
                for (question_id,) in pending_questions:
                    logger.info(f"Processing pending question ID: {question_id}")
                    processor.process_question(question_id)
            else:
                # Define the main question
                question_text = """**Objective**: Create an elaborate complete markdown report detailing advancements in artificial intelligence only in November 2024, covering hardware, software, open-source developments and emerging trends (focus multiple large companies have shown recently) from reputable and credible sources. Ensure the report highlights recent trends and innovations that reflect the latest industry shifts and focus areas. Each statement should include online references to credible sources. Structure the report to appeal to a broad audience, including both technical and strategic stakeholders. Each section should be engaging, visual, and supported by concrete data from authoritative sources, preferably the official announcements, technical documentation, or product pages of the service providers or manufacturers.
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
                logger.info(f"Main question: {question_text}")

                # Insert the main question into the database
                main_question_id = database.insert_question(conn, question_text)

                # Process the main question
                processor.process_question(main_question_id)

        except Exception as e:
            logger.exception("An error occurred during main execution.")
            print(f"An error occurred: {e}")
        finally:
            conn.close()
            logger.debug("Database connection closed.")

if __name__ == '__main__':
    main()
