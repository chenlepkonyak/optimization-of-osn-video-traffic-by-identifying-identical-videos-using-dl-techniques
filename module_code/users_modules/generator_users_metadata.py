import uuid
import random
import csv
import os
import sqlite3
import sqlite3
import os
import random
import pathlib as Path
from module_code import database_modules
from module_code.database_modules.create_model_database import *
from module_code.database_modules.db_operation import *
from module_code import model_modules
from module_code.model_modules.generator_UHVID_data import *
from module_code.model_modules.generate_dataframe import *
from module_code.model_modules.generate_video_sharing_traffic import *

class GeneratorUsersMetadata:

	def __init__(self):
		print("Generating user's metadata initiated......\n")
		pass

	def generatorUsersMetadata():	
		
		user_name = " "
		connection = sqlite3.connect('sqlite_model.db')
		mycursor = connection.cursor()

		i = 0
		tempUsers = 0

		query =  '''CREATE TABLE IF NOT EXISTS users_metadata_db (
					user_id INTEGER PRIMARY KEY AUTOINCREMENT,
					user_name VARCHAR(45) DEFAULT NULL,
					mac_id VARCHAR(50),
					mno_id INTEGER DEFAULT NULL,
					mno_name VARCHAR(45) DEFAULT NULL,
					plmn_id INTEGER DEFAULT NULL,
					plmn_name VARCHAR(45) DEFAULT NULL,
					UNIQUE(user_id)
				)'''

		mycursor.execute(query)
		connection.commit()
		
		# Fetch the number of existing users
		mycursor.execute('SELECT * FROM users_metadata_db')
		rows = mycursor.fetchall()
		tempUsers = len(rows)

		print(f"The no. of users in the system is {tempUsers} : Enter the no. of users that is to be added: ")
		no_of_users = int(input("No. of users to be added in the system: "))

		while i < no_of_users:
			macid = random.randint(111111111111111, 999999999999999)
			# macid = 123658568936176 #Real MACID
			mac_hex = hex(macid).replace('0x', '').upper()
			Formatted_MACID = '-'.join(mac_hex[i: i + 2] for i in range(0, 11, 2))
			tempUsers += 1
			user_name = "User" + str(tempUsers) + "_" + str(macid)
			directory = user_name
			
			parent_dir = "videos_directory"            
			path = os.path.join(parent_dir, directory)
			os.makedirs(path, exist_ok=True)		
			mno_id = random.randint(1, 99)
			mno_name = "mno_" + str(mno_id)
			plmn_id = random.randint(100, 300)
			plmn_name = "plmn_" + str(plmn_id)
			#userDataframe = {'user_name' : user_name, 'mno_id' : mno_id, 'mno_name' : mno_name, 'plmn_id' : plmn_id, 'plmn_name' : plmn_name }

			#print(f"Load the video datasets in the video_path: {}")
			CreateModelDatabase.createModelDatabaseForUser(user_name)
			connection = sqlite3.connect('sqlite_model.db')
			mycursor = connection.cursor()
			query = '''
					INSERT INTO users_metadata_db (user_name, mac_id, mno_id, mno_name, plmn_id, plmn_name)
					VALUES (?, ?, ?, ?, ?, ?)
				'''
			mycursor.execute(query, (user_name, Formatted_MACID, mno_id, mno_name, plmn_id, plmn_name))
			connection.commit()            
			i += 1
		connection.close()