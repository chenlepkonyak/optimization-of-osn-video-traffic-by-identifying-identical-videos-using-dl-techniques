
import socket
import random
import json
import sqlite3
import os
import sys
import numpy as np
from pathlib import Path
import threading
import time
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from module_code import database_modules
from module_code.database_modules.create_model_database import *
from module_code.database_modules.db_operation import *
from module_code import model_modules
from module_code.model_modules.generator_UHVID_data import *
from module_code.model_modules.generate_dataframe import *
from module_code.model_modules.generate_video_sharing_traffic import *
from module_code import users_modules
from module_code.users_modules.generate_server_users_metadata import *
from module_code.users_modules.generator_users_metadata import *
from module_code import utils_modules
from module_code.utils_modules.create_users_view import *
from module_code.utils_modules.generate_CSV import *
from module_code.utils_modules.display_tables_and_views import *
from module_code import videos_modules
from module_code.videos_modules.generator_server_videos_metadata import *
from module_code.videos_modules.generator_users_videos_metadata import *



class ClientModel:
    def __init__(self, host='127.0.0.1', port=9982):
        self.host = host
        self.port = port
        print("Client Model initiated.....\n")

    def send_shutdown_signal(self):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))

            # Send the shutdown signal
            client_socket.send("SHUTDOWN".encode('utf-8'))
            response = client_socket.recv(4096)

        except socket.error as e:
            print(f"Socket error: {e}")
        finally:
            client_socket.close()

    def clientmodel(self):
        retries = 5

        i = 0
        no_of_initiations = int(input("Enter No. of initiations for the traffic: "))
        while i < no_of_initiations:
            connection = sqlite3.connect('sqlite_model.db')
            mycursor = connection.cursor()
            query = "SELECT user_name FROM server_users_metadata_db"
            mycursor.execute(query)
            randomlyUsersPointer = [item[0] for item in mycursor.fetchall()]
            connection.commit()
            connection.close()

            randomlySelectedUser = random.choice(randomlyUsersPointer)

            allVideosFilesList = []
            for path, subdirs, files in os.walk(Path(f"videos_directory/{randomlySelectedUser}")):
                for name in files:
                    fileName = os.path.basename(name)
                    allVideosFilesList.append(fileName)

            randomlySelectedVideoFileName = random.choice(allVideosFilesList)

            receiversUsersTuple = [user for user in randomlyUsersPointer if user != randomlySelectedUser]
            nReceivers = random.randint(1, 5)
            selectRandomRList = np.random.choice(receiversUsersTuple, nReceivers, replace=False)
            count = len(selectRandomRList)

            selectRandomRNamesList = [selectRandomRList[x] for x in range(count)]

            if count > 1:
                modeOfSending = "Group"
                mode = 1
            elif count == 1:
                modeOfSending = "Single"
                mode = 0

            for receiver in selectRandomRNamesList:
                application = random.choice(['WhatsApp', 'Instagram', 'Facebook', 'Wechat'])

                # vidoe sharing traffic
                df = GenerateVideoSharingTraffic.videotraffic(randomlySelectedUser, randomlySelectedVideoFileName, receiver, application, count, mode)
                attempt = 0
                DBoperation.usersPushNotificationsDB(randomlySelectedUser, df)

                try:
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client_socket.connect((self.host, self.port))
                    client_socket.send(json.dumps(df).encode('utf-8'))
                    response = client_socket.recv(4096)

                    client_socket.close()
                except ConnectionRefusedError:

                    time.sleep(2)  # Wait before retrying
                except BrokenPipeError:

                    client_socket.close()

            i += 1
            print(f"Initiations:",i)
            if i == no_of_initiations:
                selectRandomRNamesList = []
                print("Traffic generation successfully completed")

# Start the client in a separate thread
client = ClientModel()
client_thread = threading.Thread(target=client.clientmodel)
client_thread.start()

# Wait for the client to complete execution
client_thread.join()