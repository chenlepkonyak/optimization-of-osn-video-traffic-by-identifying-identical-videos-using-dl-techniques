import os
import time
import glob
import random
import csv
import ffmpeg
import sqlite3
import datetime
from moviepy.video.io.VideoFileClip import VideoFileClip 
import subprocess as sp
import subprocess
import json
from module_code import database_modules
from module_code.database_modules.create_model_database import *
from module_code.database_modules.db_operation import *
from module_code import model_modules
from module_code.model_modules.generator_UHVID_data import *
from module_code.model_modules.generate_dataframe import *
from module_code.model_modules.generate_video_sharing_traffic import *
import pathlib


class GeneratorServerVideosMetadata:
    def __init__(self):
        print("Generating Server's video metadata initiated.....\n")
        pass
    
    @staticmethod
    def generatorServerVideosMetadata():
        
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        print("Please wait generating Video metadata from the systems......\n")

        flag = 0
        root_dir = Path("videos_directory")              
        mycursor.execute("SELECT * FROM server_users_metadata_db;")
        results = mycursor.fetchall()


        if not os.path.exists(root_dir): 
            print(f" Video folder '{root_dir}' does not exist. Make a video directory '{root_dir}' for {len(results)} users and load with videos for sharing ")
            exit()
        else:

            for path, subdirs, files in os.walk(root_dir):
                #print(root_dir)
                for name in files:
                    print(f"Genreated {name} video metada \n")

                if (flag==0):
                        # Check if the filename already exists in the database
                        mycursor.execute('SELECT 1 FROM server_videos_metadata_db WHERE video_name = ?', (name,))

                if mycursor.fetchone() is None:
                        flag = 1

                if flag ==1:
                        # Insert the new row
                        file_path = os.path.join(path, name)
                        print(file_path)
                        location = file_path
                        video_name = name

                        video_size = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        video = VideoFileClip(file_path)
                        video_duration = int(video.duration) 
                        video_uhvid, uhvid_size = GeneratorUHVIDdata().generatorUHVIDdata(os.path.join(path, name))                    
                        uhvid_size = round(uhvid_size, 2)
                        modTimesinceEpoc = os.path.getmtime(file_path)
                        timestamp = datetime.datetime.fromtimestamp(modTimesinceEpoc).strftime('%Y-%m-%d %H:%M:%S')

                        videoDataFrame = {
                            'location': location,
                            'video_name': video_name,
                            'video_size': video_size,
                            'video_duration': video_duration,
                            'video_uhvid': video_uhvid,
                            'uhvid_size': uhvid_size,
                            'timestamp': timestamp
                        }
                        
                        columns = ', '.join(f'"{str(x).replace("/", "_")}"' for x in videoDataFrame.keys())
                        values = ', '.join(f'"{str(x).replace("/", "_")}"' for x in videoDataFrame.values())
                        query = f'INSERT INTO server_videos_metadata_db ({columns}) VALUES ({values});'
                        mycursor.execute(query)

                else:
                        print(f"Filename  already exists in the database.") #{name}


                connection.commit()

        connection.close()
        print("Server Video Metadata successfully generated")