import sqlite3
import numpy
import os
from pathlib import Path
import datetime
from moviepy.video.io.VideoFileClip import VideoFileClip 
from module_code import model_modules
from module_code.model_modules.generator_UHVID_data import *
from module_code.model_modules.generate_dataframe import *
from module_code.model_modules.generate_video_sharing_traffic import *
from module_code import utils_modules
from module_code.utils_modules.create_users_view import *
from module_code.utils_modules.generate_CSV import *

class GeneratorUsersLocalStoreDB:
    def __init__(self):
        print("Generating Local's user datastore initiated.....\n")
        pass

    @staticmethod
    def generatorUsersLocalStoreDB():
        selectedUserList = []
        connection = sqlite3.connect('/sqlite_model.db')
        mycursor = connection.cursor()
        query = 'SELECT user_name FROM server_users_metadata_db'
        mycursor.execute(query)

        usersPointer = [item[0] for item in mycursor.fetchall()]

        for randomUserTuple in usersPointer:
            selectedUserList.append(randomUserTuple)

        for user in selectedUserList:

          for path, subdirs, files in os.walk(Path(f"/video_dataset_directory/{user}")):
              for name in files:
                  print(name)

                  file_path = os.path.join(path, name)
                  location = file_path
                  video_name = name
                  category = 0
                  video_size = round(os.path.getsize(file_path) / (1024 * 102), 2)                 

                  mycursor.execute('''SELECT video_uhvid, uhvid_size,video_duration FROM server_videos_metadata_db WHERE video_name = ?''', (name,))
                  data = mycursor.fetchall()

                  for row in data:
                    video_uhvid = row[0]
                    uhvid_size = round(row[1],2)
                    video_duration = row[2]

                  modTimesinceEpoc = os.path.getmtime(file_path)
                  timestamp = datetime.datetime.fromtimestamp(modTimesinceEpoc).strftime('%Y-%m-%d %H:%M:%S')

                  videoDataFrame = {
                      'location': location,
                      'category': category,
                      'video_name': video_name,
                      'video_size': video_size,
                      'video_duration': video_duration,
                      'video_uhvid': video_uhvid,
                      'uhvid_size': uhvid_size,
                      'timestamp': timestamp,

                  }                  

                  columns = ', '.join(f'"{str(x).replace("/", "_")}"' for x in videoDataFrame.keys())
                  values = ', '.join(f'"{str(x).replace("/", "_")}"' for x in videoDataFrame.values())
                  query = f'INSERT INTO {user}_local_store_db ({columns}) VALUES ({values});'

                  mycursor.execute(query)
                  connection.commit()

        connection.close()
        print("Users Local store video metadata updated successfully.")