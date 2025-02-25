import socket
import threading
import os
import json
import pickle
import sqlite3
import pandas as pd
import datetime

from project_source_code import database_modules
from project_source_code.database_modules.create_model_database import *
from project_source_code.database_modules.db_operation import *

class DBoperation:
    def __init__(self):
        print("DBoperation successfully initiated")
  
    @staticmethod
    def usersPushNotificationsDB(user, df):

        userDatabase = user + "_push_notification_db"
        connection = sqlite3.connect('/sqlite_model.db')
        columns = ', '.join(f'"{str(x).replace("/", "_")}"' for x in df.keys())
        values = ', '.join(f'"{str(x).replace("/", "_")}"' for x in df.values())
        query = f'INSERT INTO {userDatabase} ({columns}) VALUES ({values});'
        mycursor = connection.cursor()
        mycursor.execute(query)
        connection.commit()       
        connection.close()

    @staticmethod
    def updateLocalStore(df):
        connection = sqlite3.connect('/sqlite_model.db')
        video_name = df.get('video_name')
        user = df.get('receiver')
        receiver_application = df.get('receiver_application')
        location = f"/video_dataset_directory/{user}/{receiver_application}/{video_name}"
        category = df.get('mode')

        video_size = df.get('size')
        video_duration = df.get('duration')
        video_uhvid = df.get('video_uhvid')
        uhvid_size = df.get('uhvid_size')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        videoDataFrame = {
            'location': location,
            'category': category,
            'video_name': video_name,
            'video_size': video_size,
            'video_duration': video_duration,
            'video_uhvid': video_uhvid,
            'uhvid_size': uhvid_size,
            'timestamp': timestamp,
            'application': receiver_application
        }

        columns = ', '.join(f'"{str(x).replace("/", "_")}"' for x in videoDataFrame.keys())
        values = ', '.join(f'"{str(x).replace("/", "_")}"' for x in videoDataFrame.values())

        query = f'INSERT INTO {user}_local_store_db ({columns}) VALUES ({values});'     
        mycursor = connection.cursor()
        mycursor.execute(query)
        connection.commit()

    def displayTableContent():
        try:
            conn = sqlite3.connect('sqlite_model.db')
            cursor = conn.cursor()
            #cursor.execute("SELECT * FROM server_videos_metadata_db;")
            #cursor.execute("SELECT * FROM User31_169822377037225_ReceiverPerspective;")
            #cursor.execute("SELECT * FROM User2_113129666386352_local_store_db;")
            #cursor.execute("SELECT * FROM User42_403486510076827_local_store_db;")
            #cursor.execute("SELECT * FROM server_all_traffic_db;")
            cursor.execute("SELECT * FROM users_metadata_db;")
            #cursor.execute("SELECT * FROM server_users_metadata_db;")
            #cursor.execute("SELECT * FROM server_push_notification_accepted_db;")
            #cursor.execute("SELECT * FROM server_push_notification_deferred_db;")
            #cursor.execute("SELECT * FROM User2_113129666386352_push_notification_db;")
            results = cursor.fetchall()

            print(f"Details of users in teh user's database: {len(results)}")
            for row in results:
                print(row)

            cursor = conn.cursor()
            #cursor.execute("SELECT * FROM server_videos_metadata_db;")
            #cursor.execute("SELECT * FROM User31_169822377037225_ReceiverPerspective;")
            #cursor.execute("SELECT * FROM User2_113129666386352_local_store_db;")
            #cursor.execute("SELECT * FROM User42_403486510076827_local_store_db;")
            #cursor.execute("SELECT * FROM server_all_traffic_db;")
            cursor.execute("SELECT * FROM server_users_metadata_db;")
            #cursor.execute("SELECT * FROM server_push_notification_accepted_db;")
            #cursor.execute("SELECT * FROM server_push_notification_deferred_db;")
            #cursor.execute("SELECT * FROM User2_113129666386352_push_notification_db;")
            results = cursor.fetchall()
            
            print(f"Details of users in the server database: {len(results)}")
            for row in results:
                print(row)
            conn.close()


        except sqlite3.OperationalError as e:
            print(".")

    import sqlite3

    def GetAllTablesAndViews(db_path):
        #Retrieve all table and view names from the database
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]

            # Get all view names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%';")
            views = [row[0] for row in cursor.fetchall()]

            conn.close()

            return tables, views

        except sqlite3.Error as e:
            print(f"An error occurred while retrieving tables and views: {e}")
            return [], []

    def DropAllTablesAndViews(db_path):
        #Drop all tables and views in the database.
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all tables and views
            tables, views = DBoperation.GetAllTablesAndViews(db_path)

            # Drop all views
            for view in views:
                cursor.execute(f"DROP VIEW IF EXISTS {view}")

            # Drop all tables
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")

            conn.commit()
            conn.close()

            print("All tables and views have been dropped successfully.")

        except sqlite3.Error as e:
            print(f"An error occurred while dropping tables and views: {e}")