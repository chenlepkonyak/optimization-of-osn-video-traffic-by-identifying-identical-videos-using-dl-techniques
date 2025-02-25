import os
import time
import os.path
import glob
import random
from moviepy.video.io.VideoFileClip import VideoFileClip 
import csv
import datetime
import numpy as np
import json
from pathlib import Path
import sqlite3
from project_source_code import model_modules
from project_source_code.model_modules.generator_UHVID_data import *
from project_source_code.model_modules.generate_dataframe import *
from project_source_code.model_modules.generate_video_sharing_traffic import *
from project_source_code import utils_modules
from project_source_code.utils_modules.create_users_view import *
from project_source_code.utils_modules.generate_CSV import *

class GenerateVideoSharingTraffic:
    def __init__(self):
        self.video_id = 0
        self.file_path = " "        
        self.name = " "
        self.video_size = 0.0
        self.video_duration = 0.0
        self.uhvid_data = json.dumps(" ")
        self.uhvid_size = 0.0
        self.requestedTimeStamp = datetime.datetime.now()
        self.application = " "
        self.dataFrame = {}
        print("Video Sharing Traffic process sucessfully initiated")


    def videotraffic(sender, videoFileName, receiver, r_application, count, mode):
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()
       
        userDatabase = sender+"_local_store_db"

        for path, subdirs, files in os.walk("videos_directory"):            
            for name in files:
                if name == videoFileName:
                    file_path = os.path.join(path, name)

                    if "WhatsApp" in file_path:
                        s_application = "WhatsApp"
                    elif "Instagram" in file_path:
                        s_application = "Instagram"
                    elif "Facebook" in file_path:
                        s_application = "Facebook"
                    elif "WeChat" in file_path:
                        s_application = "WeChat"

                    (file, ext) = os.path.splitext(name)
                    video_name = name
                    video_size = os.path.getsize(file_path) / (1024 * 1024)
                    video_size = round(video_size,2)
                    query = "SELECT  video_uhvid, uhvid_size, video_duration FROM %s  where video_name = '%s'" %(userDatabase, videoFileName)
                    mycursor.execute(query)
                    records = mycursor.fetchone()
                    uhvid_data = str(records[0])
                    uhvid_size = float(records[1])
                    video_duration = int(records[2])
                    """uhvid_data, uhvid_size = GeneratorUHVIDdata().generatorUHVIDdata(os.path.join(path, name))
                    uhvid_size = round(uhvid_size, 2)"""
                    requestedTimeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        dataFrame = { 
                      'sender': sender,
                      'mode': mode,
                      'video_name': video_name,
                      'size': video_size,
                      'video_uhvid': uhvid_data,
                      'uhvid_size' : uhvid_size,
                      'duration': video_duration,
                      'receiver': receiver,
                      'relocation_timestamp': requestedTimeStamp,
                      'mem_usage_status': video_size,
                      'data_usage_status': video_size,
                      'sender_application': s_application,
                      'receiver_application': r_application
        }

        connection.close()
        return dataFrame