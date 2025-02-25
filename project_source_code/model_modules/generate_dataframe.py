import os
import time
import os.path
import glob
import random
from moviepy.video.io.VideoFileClip import VideoFileClip 
import csv
import sqlite3
import datetime
import numpy as np
import json
from pathlib import Path
from project_source_code import model_modules
from project_source_code.model_modules.generator_UHVID_data import *


class GenerateDataFrame:

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
        print("Generating DataFrame for video sharing sucessfully initiated")

    @staticmethod
    def generateDataFrame(sender, videoFileName, receiver, r_application, count, mode):   

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

                    video_name = name
                    video_size = os.path.getsize(file_path) / (1024 * 1024)
                    video_size = round(video_size, 2)

                    video = VideoFileClip(file_path)
                    video_duration = int(video.duration)

                    uhvid_data, uhvid_size = GeneratorUHVIDdata().generatorUHVIDdata(os.path.join(path, name))
                    uhvid_size = round(uhvid_size, 2)
                    requestedTimeStamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    dataFrame = {                        
                        'sender': sender,
                        'mode': mode,
                        'video_name': video_name,
                        'size': video_size,
                        'video_uhvid': uhvid_data,
                        'uhvid_size': uhvid_size,
                        'duration': video_duration,
                        'receiver': receiver,
                        'relocation_timestamp': requestedTimeStamp,
                        'mem_usage_status': video_size,
                        'data_usage_status': video_size,
                        'sender_application': s_application,
                        'receiver_application': r_application
                    }                 

                    return dataFrame       
        return None