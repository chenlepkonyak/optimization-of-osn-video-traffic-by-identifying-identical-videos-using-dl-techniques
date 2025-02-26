
# main.py
import sys
import os
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
from module_code import visualize_modules
from module_code.visualize_modules.generator_UHVID_data_visualize import *


def main():

    if len(sys.argv) > 1:
        # If an argument is provided, process only the video for visualization of uhvid generation
        video_path = sys.argv[1]
        uhvidid = GeneratorUHVIDdataVisualize().generatorUHVIDdataVisualize(video_path)
        #print(f"Generated UHVID: {uhvidid}")
        sys.exit(0) 

    sql_database_path = "sqlite_model.db"	#Replace with your own path for sqlite database
    video_path = "videos_directory"	#Replace with your own path for vidoes datasets
    csv_data_path = "csv_data_directory"	#Replace with your own path for csv data
    ######DBoperation.DropAllTablesAndViews(sql_database_path) # execute only to clean the database completely

    #CreateModelDatabase().createModelDatabase()
    #DisplayTablesAndViews.displayTablesAndViews(sql_database_path)
    #GeneratorUsersMetadata.generatorUsersMetadata()
    #GenerateServerUsersMetadata().generateServerUsersMetadata()
    DBoperation.displayTableContent()
    #GeneratorServerVideosMetadata().generatorServerVideosMetadata()
    #GeneratorUsersVideosMetadata().generatorUsersVideosMetadata()

    #Generate views and CSV
    CreateUsersView.generateviews()
    GenerateCSV.dbtocsv()   

if __name__ == "__main__":
    main()


