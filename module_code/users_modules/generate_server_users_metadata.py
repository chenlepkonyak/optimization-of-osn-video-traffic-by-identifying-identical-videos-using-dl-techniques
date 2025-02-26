import sqlite3
import os
import random
from module_code import database_modules
from module_code.database_modules.create_model_database import *
from module_code.database_modules.db_operation import *
from module_code import model_modules
from module_code.model_modules.generator_UHVID_data import *
from module_code.model_modules.generate_dataframe import *
from module_code.model_modules.generate_video_sharing_traffic import *

class GenerateServerUsersMetadata:

    def __init__(self):
        print("Generating Server's user metadata initiated.....\n")
        pass
  
    @staticmethod
    def generateServerUsersMetadata():
    
        ################################

        try:
            # Connect to SQLite database
            conn = sqlite3.connect('sqlite_model.db')
            cursor = conn.cursor()

            # Step 1: Read data from the source table
            cursor.execute("SELECT * FROM users_metadata_db")
            rows = cursor.fetchall()  # Fetch all records

            if not rows:
                print("No records found in the source table.")
                return

            # Step 2: Get column names (assuming source and destination have the same structure)
            cursor.execute("PRAGMA table_info(users_metadata_db)")
            columns = [col[1] for col in cursor.fetchall()]
            col_names = ", ".join(columns)
            placeholders = ", ".join(["?"] * len(columns))

            # Step 3: Insert only new rows into the destination table using WHERE NOT EXISTS
            for row in rows:
                conditions = " AND ".join([f"{col} = ?" for col in columns])  # Create WHERE condition
                query = f"""
                INSERT INTO server_users_metadata_db ({col_names}) 
                SELECT {placeholders} 
                WHERE NOT EXISTS (
                    SELECT 1 FROM server_users_metadata_db WHERE {conditions}
                )
                """
                cursor.execute(query, row + row)  # Pass parameters twice for WHERE condition

            # Step 4: Commit and close the connection
            conn.commit()
            print(f"Successfully transferred {len(rows)} unique records from users_metadata_db to server_users_metadata_db.")

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")

        finally:
            conn.close()