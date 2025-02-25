import sqlite3

class CreateModelDatabase:

    @staticmethod
    def createModelDatabase():
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect('sqlite_model.db')
            mycursor = conn.cursor()
            # Execute a simple query to check the connection
            mycursor.execute("SELECT sqlite_version();")
            version = mycursor.fetchone()

            print(f"Connected to SQLite successfully! Version: {version[0]}")           

            query = '''
                        CREATE TABLE IF NOT EXISTS server_videos_metadata_db (
                            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            location TEXT,
                            video_name TEXT NOT NULL,
                            video_size REAL NOT NULL,
                            video_duration INTEGER NOT NULL,
                            video_uhvid BLOB NOT NULL,
                            uhvid_size REAL DEFAULT NULL,
                            timestamp DATETIME NOT NULL
                        )
                        '''
            mycursor.execute(query)


            query = '''
                        CREATE TABLE IF NOT EXISTS server_all_traffic_db (
                            packet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            sender TEXT NOT NULL,
                            mode INTEGER,
                            video_name TEXT NOT NULL,
                            size REAL NOT NULL,
                            video_uhvid TEXT NOT NULL,
                            uhvid_size REAL DEFAULT NULL,
                            duration INTEGER DEFAULT NULL,
                            receiver TEXT NOT NULL,
                            relocation_timestamp DATETIME DEFAULT NULL,
                            mem_usage_status REAL DEFAULT NULL,
                            data_usage_status REAL DEFAULT NULL,
                            sender_application TEXT DEFAULT NULL,
                            receiver_application TEXT DEFAULT NULL
                        )
                        '''
            mycursor.execute(query)

            query = '''
                        CREATE TABLE IF NOT EXISTS server_push_notification_accepted_db (
                            packet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            sender TEXT NOT NULL,
                            mode INTEGER,
                            video_name TEXT NOT NULL,
                            size REAL NOT NULL,
                            video_uhvid TEXT NOT NULL,
                            uhvid_size REAL DEFAULT NULL,
                            duration INTEGER DEFAULT NULL,
                            receiver TEXT NOT NULL,
                            relocation_timestamp DATETIME DEFAULT NULL,
                            mem_usage_status REAL DEFAULT NULL,
                            data_usage_status REAL DEFAULT NULL,
                            sender_application TEXT DEFAULT NULL,
                            receiver_application TEXT DEFAULT NULL
                        )
                        '''
            mycursor.execute(query)

            query = '''
                        CREATE TABLE IF NOT EXISTS server_push_notification_deferred_db (
                            packet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            sender TEXT NOT NULL,
                            mode INTEGER,
                            video_name TEXT NOT NULL,
                            size REAL NOT NULL,
                            video_uhvid TEXT NOT NULL,
                            uhvid_size REAL DEFAULT NULL,
                            duration INTEGER DEFAULT NULL,
                            receiver TEXT NOT NULL,
                            relocation_timestamp DATETIME DEFAULT NULL,
                            mem_usage_status REAL DEFAULT NULL,
                            data_usage_status REAL DEFAULT NULL,
                            sender_application TEXT DEFAULT NULL,
                            receiver_application TEXT DEFAULT NULL
                        )
                        '''
            mycursor.execute(query)
        except sqlite3.Error as e:
            print(e)
            pass

        finally:
            if conn:
                conn.close()
                print("Database successfully created.")


    
    @staticmethod
    def createModelDatabaseForUser(user):
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect('sqlite_model.db')
            mycursor = conn.cursor()
            # Execute a simple query to check the connection
            mycursor.execute("SELECT sqlite_version();")
            version = mycursor.fetchone()

            print(f"Connected to SQLite successfully! Version: {version[0]}")

            query = '''
            CREATE TABLE IF NOT EXISTS users_metadata_db (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT DEFAULT NULL,
                mac_id TEXT,
                mno_id INTEGER DEFAULT NULL,
                mno_name TEXT DEFAULT NULL,
                plmn_id INTEGER DEFAULT NULL,
                plmn_name TEXT DEFAULT NULL,
                UNIQUE(user_id)
            )
            '''
            mycursor.execute(query)

            query = '''
            CREATE TABLE IF NOT EXISTS server_users_metadata_db (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT DEFAULT NULL,
                mac_id TEXT,
                mno_id INTEGER DEFAULT NULL,
                mno_name TEXT DEFAULT NULL,
                plmn_id INTEGER DEFAULT NULL,
                plmn_name TEXT DEFAULT NULL,
                UNIQUE(user_id)
            )
            '''
            mycursor.execute(query)

            query = '''
            CREATE TABLE IF NOT EXISTS server_videos_metadata_db (
                video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT,
                video_name TEXT NOT NULL,
                video_size REAL NOT NULL,
                video_duration INTEGER NOT NULL,
                video_uhvid BLOB NOT NULL,
                uhvid_size REAL DEFAULT NULL,
                timestamp DATETIME NOT NULL
            )
            '''
            mycursor.execute(query)

            query = '''
            CREATE TABLE IF NOT EXISTS server_all_traffic_db (
                packet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                mode INTEGER,
                video_name TEXT NOT NULL,
                size REAL NOT NULL,
                video_uhvid TEXT NOT NULL,
                uhvid_size REAL DEFAULT NULL,
                duration INTEGER DEFAULT NULL,
                receiver TEXT NOT NULL,
                relocation_timestamp DATETIME DEFAULT NULL,
                mem_usage_status REAL DEFAULT NULL,
                data_usage_status REAL DEFAULT NULL,
                sender_application TEXT DEFAULT NULL,
                receiver_application TEXT DEFAULT NULL
            )
            '''
            mycursor.execute(query)

            query = '''
            CREATE TABLE IF NOT EXISTS server_push_notification_accepted_db (
                packet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                mode INTEGER,
                video_name TEXT NOT NULL,
                size REAL NOT NULL,
                video_uhvid TEXT NOT NULL,
                uhvid_size REAL DEFAULT NULL,
                duration INTEGER DEFAULT NULL,
                receiver TEXT NOT NULL,
                relocation_timestamp DATETIME DEFAULT NULL,
                mem_usage_status REAL DEFAULT NULL,
                data_usage_status REAL DEFAULT NULL,
                sender_application TEXT DEFAULT NULL,
                receiver_application TEXT DEFAULT NULL
            )
            '''
            mycursor.execute(query)

            query = '''
            CREATE TABLE IF NOT EXISTS server_push_notification_deferred_db (
                packet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                mode INTEGER,
                video_name TEXT NOT NULL,
                size REAL NOT NULL,
                video_uhvid TEXT NOT NULL,
                uhvid_size REAL DEFAULT NULL,
                duration INTEGER DEFAULT NULL,
                receiver TEXT NOT NULL,
                relocation_timestamp DATETIME DEFAULT NULL,
                mem_usage_status REAL DEFAULT NULL,
                data_usage_status REAL DEFAULT NULL,
                sender_application TEXT DEFAULT NULL,
                receiver_application TEXT DEFAULT NULL
            )
            '''
            mycursor.execute(query)

            tableName = f"{user}_local_store_db"
            query = f'''
            CREATE TABLE IF NOT EXISTS {tableName} (
                video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT,
                category INTEGER DEFAULT NULL,
                video_name TEXT NOT NULL,
                video_size REAL NOT NULL,
                video_duration INTEGER NOT NULL,
                video_uhvid BLOB NOT NULL,
                uhvid_size REAL DEFAULT NULL,
                timestamp DATETIME NOT NULL,
                application VARCHAR
            )
            '''
            mycursor.execute(query)

            tableName = f"{user}_push_notification_db"
            query = f'''
            CREATE TABLE IF NOT EXISTS {tableName} (
                packet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                mode INTEGER,
                video_name TEXT NOT NULL,
                size REAL NOT NULL,
                video_uhvid TEXT NOT NULL,
                uhvid_size REAL DEFAULT NULL,
                duration INTEGER DEFAULT NULL,
                receiver TEXT NOT NULL,
                relocation_timestamp DATETIME DEFAULT NULL,
                mem_usage_status REAL DEFAULT NULL,
                data_usage_status REAL DEFAULT NULL,
                sender_application TEXT DEFAULT NULL,
                receiver_application TEXT DEFAULT NULL
            )
            '''
            mycursor.execute(query)

        except sqlite3.Error as e:
            print(e)
            pass

        finally:
            if conn:
                conn.close()
                print(f"{user} database successfully created.")