# generator.py

import json
import os
import logging
import shelve
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
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
CACHE_FILE = os.path.join("output", 'cache.db')
MAX_ATTEMPTS = 3
MAX_DEPTH = 5

# -------------------- Helper Functions --------------------

def load_cache(cache_file: str):
    return shelve.open(cache_file)

# -------------------- Classes --------------------

class TaskManager:
    def __init__(self, conn: sqlite3.Connection, cache: shelve.Shelf):
        self.conn = conn
        self.cache = cache

    def process_task(self, task: Dict[str, Any], question_id: int, question_text: str) -> Dict[str, Any]:
        task_name = task.get("name")
        parameters = task.get("parameters", {})
        logger.debug(f"Processing task '{task_name}' with parameters {parameters}")

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
                # Delegate task execution to tasks.py
                outcome = tasks.execute_task(task_name, parameters, self.cache)
                # Insert the task with the outcome
                task_id = database.insert_task(self.conn, question_id, task_name, parameters, status='completed', outcome=outcome)
                database.update_task_status(self.conn, task_id, 'completed', outcome)
            except Exception as e:
                logger.exception(f"Error executing task '{task_name}' with parameters {parameters}: {e}")
                outcome = {'error': str(e)}
                # Insert the task with the error
                task_id = database.insert_task(self.conn, question_id, task_name, parameters, status='failed', outcome=outcome)
                database.update_task_status(self.conn, task_id, 'failed', outcome)

        # Handle task-specific logic
        return self.handle_task_outcome(task_name, outcome, parameters, question_id, question_text)

    def handle_task_outcome(self, task_name: str, outcome: Dict[str, Any], parameters: Dict[str, Any], question_id: int, question_text: str) -> Dict[str, Any]:
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
                structured_data = extract_outcome.get('outcome', {}).get('extracted_text')  # Assuming 'extracted_text' instead of 'structured_data'
                if structured_data:
                    # Summarize Text Task
                    summarize_task = {
                        'name': 'summarize_text',
                        'parameters': {'text': structured_data, 'question': question_text}
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
                            'summary': summary
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

    def generate_answer_from_context(self, context: Dict[str, Any]) -> str:
        logger.debug("Generating answer from context.")
        prompt = (
            "Based on the provided context, which includes the question, tasks with their parameters and outcomes, "
            "and sub-answers, generate an answer to the question. Ensure all sources are verified and include concrete "
            "URL references. Return ONLY valid JSON with one key: 'answer' (a string). No additional text or explanations.\n\n"
            "Example:\n"
            "```json\n"
            "{\n"
            '  "answer": "Your generated answer here."\n'
            "}\n"
            "```"
        )
        prompt += f"\nContext:\n{json.dumps(context, indent=2)}\n\nAnswer:"

        try:
            response = send_llm_request(
                prompt,
                self.cache,
                ANSWER_GENERATION_MODEL_NAME,
                OLLAMA_URL,
                expect_json=True  # Expecting a JSON response
            )
            logger.debug(f"LLM response for answer generation: {response}")

            # Validate and extract the 'answer' field from the JSON response
            answer = response.get("answer", "")
            if not isinstance(answer, str):
                logger.error("The 'answer' field is missing or not a string in the JSON response.")
                return ""
            logger.debug(f"Generated answer: {answer}")
            return answer

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            logger.error(f"LLM response was: {response}")
            return ""
        except Exception as e:
            logger.exception(f"An unexpected error occurred while generating answer from context: {e}")
            return ""

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
            logger.debug(f"Maximum question depth {MAX_DEPTH} reached. Skipping subquestion generation.")
            # Attempt to generate an answer directly
            combined_context = {
                'question': question_text,
                'tasks': [],
                'sub_answers': []
            }
            answer = self.answer_generator.generate_answer_from_context(combined_context)
            if answer:
                # Validate if the answer indicates unanswerable
                is_unanswerable = self.validate_answer(question_text, answer)
                if is_unanswerable is None:
                    logger.error("Failed to validate answer. Marking question as unanswerable.")
                    database.update_question_status(self.conn, question_id, 'unanswerable')
                    return None
                elif is_unanswerable:
                    logger.info("Answer indicates the question is unanswerable.")
                    database.update_question_status(self.conn, question_id, 'unanswerable')
                    return None
                else:
                    database.update_question_status(self.conn, question_id, 'answered', answer)
                    return answer

            logger.debug("Unable to generate answer at maximum depth.")
            database.update_question_status(self.conn, question_id, 'unanswerable')
            return None

        # Step 1: Extract keywords from the question
        keywords = self.extract_keywords(question_id, question_text)
        if keywords is None:
            logger.error("Failed to extract keywords.")
            database.update_question_status(self.conn, question_id, 'unanswerable')
            return None
        logger.debug(f"Extracted keywords: {keywords}")

        # Step 2: Evaluate if the question is focused
        is_focused = self.evaluate_question_focus(question_id, question_text)
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

            # Attempt to generate an answer from the context
            combined_context = {
                'question': question_text,
                'tasks': task_outcomes,
                'sub_answers': sub_answers
            }
            answer = self.answer_generator.generate_answer_from_context(combined_context)
            if answer:
                # Validate if the answer indicates unanswerable
                is_unanswerable = self.validate_answer(question_text, answer)
                if is_unanswerable is None:
                    logger.error("Failed to validate answer. Marking question as unanswerable.")
                    database.update_question_status(self.conn, question_id, 'unanswerable')
                    return None
                elif is_unanswerable:
                    logger.info("Answer indicates the question is unanswerable.")
                    database.update_question_status(self.conn, question_id, 'unanswerable')
                    return None
                else:
                    database.update_question_status(self.conn, question_id, 'answered', answer)
                    return answer

            logger.debug("Unable to generate answer from initial tasks, generating subquestions.")
        else:
            logger.debug("Question is not focused. Generating subquestions.")

            # Generate subquestions
            sub_questions = self.generate_subquestions(question_id, question_text)
            logger.debug(f"Generated sub-questions: {sub_questions}")

            # Process sub-questions
            for sub_question_text in sub_questions:
                sub_question_id = self.get_or_create_subquestion(sub_question_text, question_id)
                sub_answer = self.process_question(sub_question_id, attempts=attempts + 1)
                if sub_answer:
                    sub_answers.append(sub_answer)
                else:
                    sub_answers.append(f"Could not find an answer to sub-question: {sub_question_text}")

        # After processing tasks and/or subquestions, attempt to generate an answer using the collected information
        combined_context = {
            'question': question_text,
            'tasks': task_outcomes,
            'sub_answers': sub_answers
        }
        answer = self.answer_generator.generate_answer_from_context(combined_context)
        if answer:
            # Validate if the answer indicates unanswerable
            is_unanswerable = self.validate_answer(question_text, answer)
            if is_unanswerable is None:
                logger.error("Failed to validate answer. Marking question as unanswerable.")
                database.update_question_status(self.conn, question_id, 'unanswerable')
                return None
            elif is_unanswerable:
                logger.info("Answer indicates the question is unanswerable.")
                database.update_question_status(self.conn, question_id, 'unanswerable')
                return None
            else:
                database.update_question_status(self.conn, question_id, 'answered', answer)
                return answer
        else:
            # Check for pending tasks or subquestions
            if not self.has_pending_tasks_or_subquestions(question_id):
                logger.debug("No more pending tasks or subquestions, retrying.")
                return self.process_question(question_id, attempts=attempts + 1)
            else:
                logger.debug("There are still pending tasks or subquestions.")
                return None

    def validate_answer(self, question: str, answer: str) -> Optional[bool]:
        """
        Validates whether the generated answer indicates that the question is unanswerable.

        Args:
            question (str): The original question text.
            answer (str): The generated answer text.

        Returns:
            Optional[bool]: True if the answer indicates unanswerable, False if answerable, None if validation failed.
        """
        validation_task = {
            "name": "validate_answer",
            "parameters": {"question": question, "answer": answer}
        }
        outcome = self.task_manager.process_task(validation_task, question_id=None, question_text=question)
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

# -------------------- Main Execution Flow --------------------

def main():
    # Initialize the database
    conn = database.init_db()

    # Open a persistent cache with shelve
    with load_cache(CACHE_FILE) as cache:
        try:
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

            # Initialize the question processor
            processor = QuestionProcessor(conn, cache)

            # Process the main question
            main_answer = processor.process_question(main_question_id)

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

if __name__ == '__main__':
    main()
