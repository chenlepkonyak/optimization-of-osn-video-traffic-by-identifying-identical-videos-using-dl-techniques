import sqlite3
import os
import random
from project_source_code import database_modules
from project_source_code.database_modules.create_model_database import *
from project_source_code.database_modules.db_operation import *
from project_source_code import model_modules
from project_source_code.model_modules.generator_UHVID_data import *
from project_source_code.model_modules.generate_dataframe import *
from project_source_code.model_modules.generate_video_sharing_traffic import *

class GenerateServerUsersMetadata:

    def __init__(self):
        print("Generating Server's user metadata initiated.....")
        pass
  
    @staticmethod
    def generateServerUsersMetadata():
    
        """
        user_name = " "
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        i = 0
        tempUsers = 0
       
       
        query =  '''CREATE TABLE IF NOT EXISTS users_metadata_db (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name VARCHAR(45) DEFAULT NULL,
                    mac_id VARCHAR(50),
                    mno_id INTEGER DEFAULT NULL,
                    mno_name VARCHAR(45) DEFAULT NULL,
                    plmn_id INTEGER DEFAULT NULL,
                    plmn_name VARCHAR(45) DEFAULT NULL,
                    UNIQUE(user_id)
                   )'''

        mycursor.execute(query)
        connection.commit()
        
        # Fetch the number of existing users
        mycursor.execute('SELECT * FROM users_metadata_db')
        rows = mycursor.fetchall()
        tempUsers = len(rows)

        """"""
        no_of_users = int(input("No. of users to be added in the system: "))
       
        while i < no_of_users:
            macid = random.randint(111111111111111, 999999999999999)
            # macid = 123658568936176 #Real MACID
            mac_hex = hex(macid).replace('0x', '').upper()
            Formatted_MACID = '-'.join(mac_hex[i: i + 2] for i in range(0, 11, 2))

            tempUsers += 1
            user_name = "User" + str(tempUsers) + "_" + str(macid)
            directory = user_name
            parent_dir = "videos_directory"            
            path = os.path.join(parent_dir, directory)
            os.mkdir(path)
            mno_id = random.randint(1, 99)
            mno_name = "mno_" + str(mno_id)
            plmn_id = random.randint(100, 300)
            plmn_name = "plmn_" + str(plmn_id)
            #userDataframe = {'user_name' : user_name, 'mno_id' : mno_id, 'mno_name' : mno_name, 'plmn_id' : plmn_id, 'plmn_name' : plmn_name }

            #print(f"Load the video datasets in the video_path: {}")
            #CreateModelDatabase.createModelDatabaseForUser(user_name)
            connection = sqlite3.connect('sqlite_model.db')
            mycursor = connection.cursor()
            query = '''
                    INSERT INTO server_users_metadata_db (user_name, mac_id, mno_id, mno_name, plmn_id, plmn_name)
                    VALUES (?, ?, ?, ?, ?, ?)
                '''
            mycursor.execute(query, (user_name, Formatted_MACID, mno_id, mno_name, plmn_id, plmn_name))
            connection.commit()            
            i += 1
        connection.close()
        """

    
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