
# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from project_source_code import database_modules
from project_source_code.database_modules.create_model_database import *
from project_source_code.database_modules.db_operation import *
from project_source_code import model_modules
from project_source_code.model_modules.generator_UHVID_data import *
from project_source_code.model_modules.generate_dataframe import *
from project_source_code.model_modules.generate_video_sharing_traffic import *
from project_source_code import users_modules
from project_source_code.users_modules.generate_server_users_metadata import *
from project_source_code.users_modules.generator_users_local_store_DB import *
from project_source_code.users_modules.generator_users_metadata import *
from project_source_code import utils_modules
from project_source_code.utils_modules.create_users_view import *
from project_source_code.utils_modules.generate_CSV import *
from project_source_code.utils_modules.display_tables_and_views import *
from project_source_code import videos_modules
from project_source_code.videos_modules.generator_server_videos_metadata import *
from project_source_code.videos_modules.generator_users_videos_metadata import *


"""
o1 = CreateModelDatabase()
o2 = DBoperation()
o3 = GenerateDataFrame()
o4 = GenerateVideoSharingTraffic()
o5 = GeneratorUHVIDdata()
o6 = ServerModel()
o7 = GenerateServerUsersMetadata()
o8 = GeneratorUsersLocalStoreDB()
o9 = GeneratorUsersMetadata()
o10 = CreateUsersView()
o11 = DisplayTablesAndViews()
o12 = GenerateCSV()
o13 = GeneratorServerVideosMetadata()
O14 = GeneratorUsersVideosMetadata()
o15 = ClientModel()

o1 = 
o2 = 
o3 = GenerateDataFrame()
o4 = GenerateVideoSharingTraffic()
o5 = 
o6 = ServerModel()
o7 = 
o8 = 
o9 = 
o10 = 
                                    o11 = DisplayTablesAndViews()
o12 =
o13 = 
O14 = 
o15 = ClientModel()
"""

#print("Successful")


def main():

    sql_database_path = "sqlite_model.db"	#Replace with your own path for sqlite database
    video_path = "videos_directory"	#Replace with your own path for vidoes datasets
    csv_data_path = "csv_data_directory"	#Replace with your own path for csv data
    #DBoperation.DropAllTablesAndViews(sql_database_path)
    CreateModelDatabase().createModelDatabase()
    DisplayTablesAndViews.displayTablesAndViews(sql_database_path)
    GeneratorUsersMetadata.generatorUsersMetadata()
    GenerateServerUsersMetadata().generateServerUsersMetadata()
    DBoperation.displayTableContent()
    GeneratorServerVideosMetadata().generatorServerVideosMetadata()
    #GeneratorUsersVideosMetadata().generatorUsersVideosMetadata()

    """
    # Start the server in a separate thread
    server_model = ServerModel()
    server_thread = threading.Thread(target=server_model.start_server, daemon=True)
    server_thread.start()
    """
    
     

    # Generate views and CSV
    CreateUsersView.generateviews()
    GenerateCSV.dbtocsv()

    """
    CreateModelDatabase()
    GeneratorUsersLocalStoreDB()
    GenerateServerUsersMetadata()
    GeneratorUsersVideosMetadata()
    GeneratorServerVideosMetadata()
    
    
    # Start the server in a separate thread
    server_model = ServerModel()
    server_thread = threading.Thread(target=server_model.start_server, daemon=True)
    server_thread.start()

    
    # Start the client in a separate thread
    client = ClientModel()
    client_thread = threading.Thread(target=client.clientmodel)
    client_thread.start()

    # Wait for the client to complete execution
    client_thread.join()
    """
    

    # Generate views and CSV
    #CreateUsersView.generateviews()
    #GenerateCSV.dbtocsv()

if __name__ == "__main__":
    main()


