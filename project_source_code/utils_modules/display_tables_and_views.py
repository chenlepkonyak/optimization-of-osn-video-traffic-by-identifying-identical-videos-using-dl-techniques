import sqlite3


class DisplayTablesAndViews:
    def __init__(self):
        print("Displaying table and views sucessfully initiated")
        pass

    def displayTablesAndViews(database_path):
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()

            # Execute the query to get the table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            # Fetch all results
            tables = cursor.fetchall()

            # Print the table names
            for table in tables:
                print(table[0])

            cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%';")

            # Fetch all results
            tables = cursor.fetchall()

            # Print the table names
            for table in tables:
                print(table[0])

            conn.close()

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")