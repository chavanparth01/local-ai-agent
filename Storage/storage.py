# *standarad imports : 
import sqlite3
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# print("This is the root of the project", PROJECT_ROOT)

DB_PATH = PROJECT_ROOT/"assistant.db"
# print("This is the root of the project", DB_PATH)

def initialise_db():
    """create the DB table if they do not exists"""
    conn = sqlite3.connect(DB_PATH)

    # Successful, conn.commit() is called automatically afterwards by the with block
    with conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(""" 
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New conversation',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_json TEXT NOT NULL,
                conversation_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP 
            )
        """)

    # conn.rollback() is called after the with block finishes with an exception,
    conn.close()

    print(f"Database initialized at: {DB_PATH}")


def add_conversation(title="conversation title"):
    """Create a conversation and return its ID."""
    conn = sqlite3.connect(DB_PATH)

    with conn:
        conn.execute("PRAGMA foreign_keys = ON")

        cursor = conn.execute(
            "INSERT INTO conversations (title) VALUES (?)",
            (title,)
        )

        conversation_id = cursor.lastrowid

    conn.close()

    return conversation_id

def save_message(conversation_id, message):
    """Save a message dictionary to a conversation."""

    message_json = json.dumps(message)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(
            """
            INSERT INTO messages (conversation_id, message_json)
            VALUES (?, ?)
            """,
            (conversation_id, message_json)
        )
    conn.close()


def get_messages(conversation_id):
    """Retrieve messages for a conversation in insertion order."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        rows = conn.execute(
            """
            SELECT message_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,)
        ).fetchall()

    conn.close()
    return [json.loads(row[0]) for row in rows]


def get_conversations():
    with sqlite3.connect(DB_PATH) as conn:
        """Return all saved conversations, newest first."""

        rows = conn.execute(
            """SELECT id, title, created_at
                FROM conversations
                ORDER BY id DESC
            """
        ).fetchall()


    conn.close()
    return rows

def save_user_fact(fact):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO user_facts (fact) VALUES (?)", (fact,)
        )

    conn.close()

def get_user_facts():
    """Retrieve all saved user facts as a list of tuples (id, fact)."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, fact FROM user_facts ORDER BY id ASC"
        ).fetchall()
        
    return rows  

def delete_user_fact_db(fact_id):
    """Delete a user fact by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM user_facts WHERE id = ?", (fact_id,))

def update_user_fact_db(fact_id, new_fact):
    """Update an existing user fact by its ID."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE user_facts SET fact = ? WHERE id = ?",
            (new_fact, fact_id)
        )


# !CALL the functions only when this module is executed seperately
if __name__ == "__main__":
    # !Test anything you want about this file 
    initialise_db()
    # print(get_user_facts())
