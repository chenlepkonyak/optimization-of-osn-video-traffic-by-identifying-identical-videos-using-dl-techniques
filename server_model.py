import socket
import threading
import json
import sqlite3
import time
import os
import sys

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


class ServerModel:
    def __init__(self, host='0.0.0.0', port=9982):
        self.server_running = True
        self.host = host
        self.port = port
        self.server_socket = None
        self.flag = 0
        print("Server model  initiated......\n")

    def updateTrafficData(self, df):
        retry_count = 0
        max_retries = 5
        #UPDATE TRAFFIC ENTRY
        #while retry_count < max_retries:
        try:
            connection = sqlite3.connect('sqlite_model.db')
            connection.execute('PRAGMA busy_timeout = 30000')  # 30 seconds timeout
            mycursor = connection.cursor()

            columns = ', '.join(f'"{str(x).replace("/", "_")}"' for x in df.keys())
            values = ', '.join(f'"{str(x).replace("/", "_")}"' for x in df.values())
            query = f"INSERT INTO server_all_traffic_db ({columns}) VALUES ({values});"

            mycursor.execute(query)
            connection.commit()            

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                retry_count += 1
                print(f"Database is locked, retrying {retry_count}/{max_retries} after a short delay...")
                time.sleep(5)  # Delay before retrying
            else:
                print(f"OperationalError: {e}")
                #break
        except Exception as e:
            print(f"An error occurred: {e}")
            
        finally:
            if mycursor:
                mycursor.close()
            if connection:
                #"UPDATE TRAFFIC EXIT
                connection.close()


    def compareUHVIDdata(self, df):
        
        try:
            connection = sqlite3.connect('sqlite_model.db')
            mycursor = connection.cursor()

            keyUHVIDdata = df.get("video_uhvid")
            usersDatabase = df.get("receiver") + "_local_store_db"

            flag1, flag2, self.flag = 0, 0, 0

            query = "SELECT video_uhvid FROM server_videos_metadata_db"
            mycursor.execute(query)
            fd1 = mycursor.fetchall()
            for t1 in fd1:
                if keyUHVIDdata in t1:
                    flag1 = 1

            query = f"SELECT video_uhvid FROM {usersDatabase}"
            mycursor.execute(query)
            fd2 = mycursor.fetchall()
            for t2 in fd2:
                if keyUHVIDdata in t2:
                    flag2 = 1                    
            if flag1 == flag2:
              self.flag = 1
            else:
               self.flag = 0
               DBoperation.updateLocalStore(df)           
        except sqlite3.OperationalError as e:
            print(f"OperationalError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if mycursor:
                mycursor.close()
            if connection:                
                connection.close()

    def acceptedDeferredDB(self, flag, df):
        #acceptedDeferredDB ENTRY
        try:
            connection = sqlite3.connect('sqlite_model.db')
            mycursor = connection.cursor()


            if flag == 1:
                print("Deferred")
                table_name = 'server_push_notification_deferred_db'
            else:
                print("Accepted")
                table_name = 'server_push_notification_accepted_db'

            columns = ', '.join(f'"{str(x).replace("/", "_")}"' for x in df.keys())
            values = ', '.join(f'"{str(x).replace("/", "_")}"' for x in df.values())
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({values});"

            mycursor.execute(query)
            connection.commit()
          
        except sqlite3.OperationalError as e:
            print(f"OperationalError: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if mycursor:
                mycursor.close()
            if connection:
                #acceptedDeferredDB EXIT
                connection.close()

    def handle_client(self, client_socket):
        shutdown_signal = "SHUTDOWN"
        try:
            request = client_socket.recv(1024)
            request_str = request.decode('utf-8')

            #Server has received packet at handler
            if request_str == shutdown_signal:
                print("Shutdown signal received. Stopping the server...")
                client_socket.send("Server shutting down...".encode('utf-8'))
                self.server_running = False
                return

            df = json.loads(request)
            #update traffic database
            self.updateTrafficData(df)
            #compare uhv database
            self.compareUHVIDdata(df)
            #accepted or deferred database
            self.acceptedDeferredDB(self.flag, df)
            #to update traffic database
            client_socket.send("ACK that packet received and compared".encode('utf-8'))


        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except sqlite3.OperationalError as e:
            print(f"SQLite operational error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            #Closed from handler
            client_socket.close()

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server started on {self.host}:{self.port}, waiting for connections...")

        while self.server_running:
            try:
                
                client_socket, addr = self.server_socket.accept()
                client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_handler.start()
            except OSError as e:
                if not self.server_running:
                    print("Server socket closed gracefully.")
                else:
                    print(f"Unexpected error: {e}")
                    break
        print("Exited from while loop of server")
        self.server_socket.close()

# Start the server in a separate thread
server_model = ServerModel()
server_thread = threading.Thread(target=server_model.start_server, daemon=True)
server_thread.start()

# Wait for user input to stop the server
input("Press ENTER to stop the server...\n")
server_model.running = False  # Stop the server
server_thread.join()
print("Server has stopped.")