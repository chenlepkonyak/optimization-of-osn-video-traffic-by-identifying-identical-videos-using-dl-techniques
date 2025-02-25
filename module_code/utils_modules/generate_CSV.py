import sqlite3
import csv
import os
import csv
import os
import sqlite3
import datetime
import json
import time

class GenerateCSV:
  def __init__(self):
        print("Generating csv data initiated......\n")
        pass
   
  def export_query_to_csv(db_path, query, csv_folder, csv_filename):
      try:
          # Connect to the SQLite database
          conn = sqlite3.connect(db_path)
          cursor = conn.cursor()

          # Execute the SELECT query
          cursor.execute(query)

          # Fetch all rows from the executed query
          rows = cursor.fetchall()

          # Get column names from the cursor description
          column_names = [description[0] for description in cursor.description]

          # Ensure the folder exists
          os.makedirs(csv_folder, exist_ok=True)

          # Construct the full path for the CSV file
          csv_file_path = os.path.join(csv_folder, f"{csv_filename}.csv")

          # Write the query results to a CSV file
          with open(csv_file_path, 'w', newline='') as csv_file:
              csv_writer = csv.writer(csv_file)

              # Write the column headers
              csv_writer.writerow(column_names)

              # Write the data rows
              csv_writer.writerows(rows)

          # Close the database connection
          conn.close()        
      except sqlite3.Error as e:
          print(f"An error occurred: {e}")

  def list_and_export_tables(db_path, csv_folder):
      try:
          conn = sqlite3.connect(db_path)
          cursor = conn.cursor()

          # Get the table names
          cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
          tables = cursor.fetchall()

          # Iterate over the table names and export each one to CSV
          for table in tables:
              table_name = table[0]
              query = f"SELECT * FROM {table_name};"
              GenerateCSV.export_query_to_csv(db_path, query, csv_folder, table_name)


          conn.close()

      except sqlite3.Error as e:
          print(f"An error occurred: {e}")

  def list_and_export_views(db_path, csv_folder):
      try:
          conn = sqlite3.connect(db_path)
          cursor = conn.cursor()

          # Get the table names
          cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%';")
          views = cursor.fetchall()

          # Iterate over the table names and export each one to CSV
          for view in views:
              view_name = view[0]
              query = f"SELECT * FROM {view_name};"
              GenerateCSV.export_query_to_csv(db_path, query, csv_folder, view_name)
          conn.close()
      except sqlite3.Error as e:
          print(f"An error occurred: {e}")

  def dbtocsv():        
    db_path = 'sqlite_model.db'
    csv_folder = 'csv_data_directory'
    GenerateCSV.list_and_export_tables(db_path, csv_folder)
    GenerateCSV.list_and_export_views(db_path, csv_folder)
    #print("CSV file generation sucessfully completed.")