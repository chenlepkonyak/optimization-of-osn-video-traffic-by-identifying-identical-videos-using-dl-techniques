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
from project_source_code import database_modules
from project_source_code.database_modules.create_model_database import *
from project_source_code.database_modules.db_operation import *
from project_source_code import model_modules
from project_source_code.model_modules.generator_UHVID_data import *
from project_source_code.model_modules.generate_dataframe import *
from project_source_code.model_modules.generate_video_sharing_traffic import *
import pathlib


class GeneratorServerVideosMetadata:
    def __init__(self):
        print("Generating Server's video metadata sucessfully initiated")
        pass

    """
    @staticmethod
    def get_video_duration(file_path):
        
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            file_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       
        try:
            output = result.stdout.decode('utf-8')
        except UnicodeDecodeError:
            output = result.stdout.decode('utf-8', errors='ignore')

        info = json.loads(output)
        duration = float(info['format']['duration'])

        return duration
    """

    @staticmethod
    def generatorServerVideosMetadata():
        
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        print("Please wait generating Video metadata from the systems....")

        flag = 0
        root_dir = Path("videos_directory")
        for path, subdirs, files in os.walk(root_dir):
            print(root_dir)
            for name in files:
              print(f"Hello {name} here genreated")

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
                    
                    #video_duration = GeneratorServerVideosMetadata().get_video_duration(file_path)

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