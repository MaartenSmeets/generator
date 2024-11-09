# database.py

import sqlite3
import json
import os
import logging
from typing import Optional, Dict, Any, List

# -------------------- Logging Configuration --------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure that logging handlers are not duplicated
if not logger.handlers:
    # Define the output directory
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define the path for the log file
    LOG_FILE = os.path.join(OUTPUT_DIR, 'database.log')
    
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

DATABASE_FILE = os.path.join("output", 'tasks.db')

# -------------------- Database Functions --------------------

def init_db() -> sqlite3.Connection:
    """
    Initialize the SQLite database with necessary tables.

    Returns:
        sqlite3.Connection: The database connection object.
    """
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
    logger.debug("Database initialized with tables 'Questions' and 'Tasks'.")
    return conn

def insert_question(conn: sqlite3.Connection, text: str, parent_id: Optional[int] = None, status: str = 'pending', answer: Optional[str] = None) -> int:
    """
    Insert a new question into the database or return existing question ID.

    Args:
        conn (sqlite3.Connection): The database connection object.
        text (str): The text of the question.
        parent_id (Optional[int], optional): The ID of the parent question. Defaults to None.
        status (str, optional): The status of the question. Defaults to 'pending'.
        answer (Optional[str], optional): The answer to the question. Defaults to None.

    Returns:
        int: The ID of the inserted or existing question.
    """
    cursor = conn.cursor()
    # Check if the question already exists
    cursor.execute('SELECT id FROM Questions WHERE text = ? AND parent_id IS ?', (text, parent_id))
    existing_question = cursor.fetchone()
    if existing_question:
        logger.debug(f"Question already exists with ID: {existing_question[0]}")
        return existing_question[0]
    cursor.execute('''
        INSERT INTO Questions (text, parent_id, status, answer)
        VALUES (?, ?, ?, ?)
    ''', (text, parent_id, status, answer))
    conn.commit()
    inserted_id = cursor.lastrowid
    logger.debug(f"Inserted new question with ID: {inserted_id}")
    return inserted_id

def update_question_status(conn: sqlite3.Connection, question_id: int, status: str, answer: Optional[str] = None):
    """
    Update the status and answer of a question.

    Args:
        conn (sqlite3.Connection): The database connection object.
        question_id (int): The ID of the question to update.
        status (str): The new status of the question.
        answer (Optional[str], optional): The answer to the question. Defaults to None.
    """
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Questions SET status = ?, answer = ? WHERE id = ?
    ''', (status, answer, question_id))
    conn.commit()
    logger.debug(f"Updated question ID {question_id} to status '{status}' with answer '{answer}'.")

def insert_task(conn: sqlite3.Connection, question_id: int, task_type: str, parameters: Dict[str, Any], status: str = 'pending', outcome: Optional[Dict[str, Any]] = None) -> int:
    """
    Insert a new task into the database or return existing task ID.

    Args:
        conn (sqlite3.Connection): The database connection object.
        question_id (int): The ID of the associated question.
        task_type (str): The type/name of the task.
        parameters (Dict[str, Any]): The parameters required for the task.
        status (str, optional): The status of the task. Defaults to 'pending'.
        outcome (Optional[Dict[str, Any]], optional): The outcome/result of the task. Defaults to None.

    Returns:
        int: The ID of the inserted or existing task.
    """
    cursor = conn.cursor()
    parameters_json = json.dumps(parameters, sort_keys=True)
    # Check if the task already exists
    cursor.execute('''
        SELECT id FROM Tasks WHERE question_id = ? AND task_type = ? AND parameters = ?
    ''', (question_id, task_type, parameters_json))
    existing_task = cursor.fetchone()
    if existing_task:
        logger.debug(f"Task already exists with ID: {existing_task[0]}")
        return existing_task[0]
    outcome_json = json.dumps(outcome) if outcome else None
    cursor.execute('''
        INSERT INTO Tasks (question_id, task_type, parameters, status, outcome)
        VALUES (?, ?, ?, ?, ?)
    ''', (question_id, task_type, parameters_json, status, outcome_json))
    conn.commit()
    inserted_id = cursor.lastrowid
    logger.debug(f"Inserted new task with ID: {inserted_id}")
    return inserted_id

def update_task_status(conn: sqlite3.Connection, task_id: int, status: str, outcome: Optional[Dict[str, Any]] = None):
    """
    Update the status and outcome of a task.

    Args:
        conn (sqlite3.Connection): The database connection object.
        task_id (int): The ID of the task to update.
        status (str): The new status of the task.
        outcome (Optional[Dict[str, Any]], optional): The outcome/result of the task. Defaults to None.
    """
    cursor = conn.cursor()
    outcome_json = json.dumps(outcome) if outcome else None
    cursor.execute('''
        UPDATE Tasks SET status = ?, outcome = ? WHERE id = ?
    ''', (status, outcome_json, task_id))
    conn.commit()
    logger.debug(f"Updated task ID {task_id} to status '{status}' with outcome '{outcome_json}'.")

def get_existing_task_outcome(conn: sqlite3.Connection, task_type: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Retrieve the outcome of an existing completed task.

    Args:
        conn (sqlite3.Connection): The database connection object.
        task_type (str): The type/name of the task.
        parameters (Dict[str, Any]): The parameters used for the task.

    Returns:
        Optional[Dict[str, Any]]: The outcome of the task if found, else None.
    """
    cursor = conn.cursor()
    parameters_json = json.dumps(parameters, sort_keys=True)
    cursor.execute('''
        SELECT outcome FROM Tasks
        WHERE task_type = ? AND parameters = ? AND status = 'completed'
    ''', (task_type, parameters_json))
    row = cursor.fetchone()
    if row and row[0]:
        outcome = json.loads(row[0])
        logger.debug(f"Retrieved existing task outcome: {outcome}")
        return outcome
    logger.debug("No existing task outcome found.")
    return None

def has_task(conn: sqlite3.Connection, question_id: int, task_type: str) -> bool:
    """
    Check if a task of a specific type has been performed for a question.

    Args:
        conn (sqlite3.Connection): The database connection object.
        question_id (int): The ID of the question.
        task_type (str): The type/name of the task.

    Returns:
        bool: True if the task exists, False otherwise.
    """
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id FROM Tasks WHERE question_id = ? AND task_type = ?
    ''', (question_id, task_type))
    existing_task = cursor.fetchone()
    if existing_task:
        logger.debug(f"Task '{task_type}' exists for question ID {question_id}.")
        return True
    else:
        logger.debug(f"Task '{task_type}' does not exist for question ID {question_id}.")
        return False

def get_task_outcome(conn: sqlite3.Connection, question_id: int, task_type: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the outcome of a specific task for a question.

    Args:
        conn (sqlite3.Connection): The database connection object.
        question_id (int): The ID of the question.
        task_type (str): The type/name of the task.

    Returns:
        Optional[Dict[str, Any]]: The outcome of the task if found, else None.
    """
    cursor = conn.cursor()
    cursor.execute('''
        SELECT outcome FROM Tasks
        WHERE question_id = ? AND task_type = ? AND status = 'completed'
    ''', (question_id, task_type))
    row = cursor.fetchone()
    if row and row[0]:
        outcome = json.loads(row[0])
        logger.debug(f"Retrieved outcome for task '{task_type}' for question ID {question_id}: {outcome}")
        return outcome
    logger.debug(f"No outcome found for task '{task_type}' for question ID {question_id}.")
    return None

def get_subquestions(conn: sqlite3.Connection, question_id: int) -> List[str]:
    """
    Retrieve the texts of subquestions for a given question ID.

    Args:
        conn (sqlite3.Connection): The database connection object.
        question_id (int): The ID of the parent question.

    Returns:
        List[str]: A list of subquestion texts.
    """
    cursor = conn.cursor()
    cursor.execute('''
        SELECT text FROM Questions WHERE parent_id = ?
    ''', (question_id,))
    subquestions = [row[0] for row in cursor.fetchall()]
    logger.debug(f"Retrieved subquestions for question ID {question_id}: {subquestions}")
    return subquestions
