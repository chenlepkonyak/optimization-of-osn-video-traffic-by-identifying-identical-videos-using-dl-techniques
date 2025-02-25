import os
import time
import glob
import random
from moviepy.video.io.VideoFileClip import VideoFileClip 
import csv
import sqlite3
import datetime
from project_source_code import database_modules
from project_source_code.database_modules.create_model_database import *
from project_source_code.database_modules.db_operation import *
from project_source_code import model_modules
from project_source_code.model_modules.generator_UHVID_data import *
from project_source_code.model_modules.generate_dataframe import *
from project_source_code.model_modules.generate_video_sharing_traffic import *

class GeneratorUsersVideosMetadata:
    def __init__(self):
        print("Generating user's video metadata initiated.....")
        pass
     
    @staticmethod
    def generatorUsersVideosMetadata():
        """
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        trafficDataCSVFileName = input('Enter a CSV Filename: ')
        fileExists = os.path.isfile(trafficDataCSVFileName)
        i = 0
        no_of_records = 10

        with open(trafficDataCSVFileName, 'a', newline='') as CSVFile:
            print("Please wait generating Video metadata from the systems....")

            csvHeader = ['VIDEO_ID', 'CATEGORY', 'LOCATION', 'VIDEO_NAME', 'VIDEO_SIZE', 'VIDEO_DURATION',
                         'VIDEO_UHVID', 'UHVID_SIZE', 'TIMESTAMP']
            writer = csv.DictWriter(CSVFile, fieldnames=csvHeader)

            if not fileExists:
                writer.writeheader()

            for path, subdirs, files in os.walk("video_directory\\"):
                for name in files:
                    if i == no_of_records:
                        break

                    file_path = os.path.join(path, name)
                    category = random.randint(0, 1)

                    (file, ext) = os.path.splitext(name)
                    video_id = file[-5:]
                    video_id = "vid_" + video_id

                    video_size = os.path.getsize(file_path) / (1024 * 102)
                    video = VideoFileClip(file_path)
                    video_duration = int(video.duration)
                    video_file = file_path
                    uhvid_data, uhvid_size = GeneratorUHVIDdata().generatorUHVIDdata(os.path.join(path, name))
                    modTimesinceEpoc = os.path.getmtime(file_path)
                    modificationTime = datetime.datetime.fromtimestamp(modTimesinceEpoc).strftime('%Y-%m-%d %H:%M:%S')

                    query = '''
                    INSERT INTO videos_metadata (video_id, category, location, video_name, video_size, video_duration, video_uhvid, uhvid_size, timestamp)
                    VALUES ('{video_id}', '{category}', '{file_path}', '{name}', '{video_size}', '{video_duration}', '{uhvid_data}', '{uhvid_size}', '{modificationTime}')
                    '''
                    mycursor.execute(query)
                    connection.commit()

                    writer.writerow({
                        'VIDEO_ID': video_id, 'CATEGORY': category, 'LOCATION': file_path, 'VIDEO_NAME': name,
                        'VIDEO_SIZE': video_size, 'VIDEO_DURATION': video_duration, 'VIDEO_UHVID': uhvid_data, 'UHVID_SIZE': uhvid_size, 'TIMESTAMP': modificationTime
                    })
                    i += 1

        mycursor.close()
        connection.close()
        print("Users Local store video metadata updated successfully.")
        """

        
        selectedUserList = []
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        query = 'SELECT user_name FROM users_metadata_db'
        mycursor.execute(query)

        usersPointer = [item[0] for item in mycursor.fetchall()]

        for randomUserTuple in usersPointer:
            selectedUserList.append(randomUserTuple)
        
        

        for user in selectedUserList:
          print(user)
          for path, subdirs, files in os.walk(Path(f"videos_directory/{user}")):
              print(path)
              for name in files:
                  print(name)
                  file_path = os.path.join(path, name)
                  location = file_path
                  video_name = name
                  category = 0
                  video_size = round(os.path.getsize(file_path) / (1024 * 102), 2)
                  video = VideoFileClip(file_path)
                  video_duration = int(video.duration)
                  video_uhvid, uhvid_size = GeneratorUHVIDdata().generatorUHVIDdata(os.path.join(path, name))

                  #mycursor.execute('''SELECT video_uhvid, uhvid_size,video_duration FROM server_videos_metadata_db WHERE video_name = ?''', (name,))
                  #data = mycursor.fetchall()
                  #for row in data:
                    #video_uhvid = row[0]
                    #uhvid_size = round(row[1],2)
                    #video_duration = row[2]

                  uhvid_size = round(uhvid_size, 2)

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