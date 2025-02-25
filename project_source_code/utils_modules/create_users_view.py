import sqlite3


class CreateUsersView:
    def __init__(self):
        print("Creating user's view sucessfully initiated")
        pass

    @staticmethod
    def createReceiversView(viewName, receiverName):

        connection = sqlite3.connect('sqlite_model.db', uri=True)
        mycursor = connection.cursor()       

        # Drop the view if it exists
        drop_view_query = f"DROP VIEW IF EXISTS {viewName}"
        mycursor.execute(drop_view_query)

        # Create the new view
        create_view_query = f" CREATE VIEW {viewName} AS SELECT * FROM server_all_traffic_db WHERE sender = '{receiverName}' ORDER BY relocation_timestamp"

        mycursor.execute(create_view_query)
        connection.commit()        
        connection.close()

    @staticmethod
    def receivers():
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        query = f"SELECT DISTINCT receiver FROM server_all_traffic_db"
        mycursor.execute(query)

        allReceivers = mycursor.fetchall()
        receiverNameList = [receiverNameTuple[0] for receiverNameTuple in allReceivers]

        for receiverName in receiverNameList:
            viewName = f"{receiverName}_ReceiverPerspective"
            CreateUsersView.createReceiversView(viewName, receiverName)
        
        connection.close()

    @staticmethod
    def createSendersView(viewName, senderName):

        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        #Drop the view if it exists
        drop_view_query = f'DROP VIEW IF EXISTS {viewName}'
        mycursor.execute(drop_view_query)

        # Create the new view'=
        create_view_query = f'''CREATE VIEW {viewName}  AS SELECT * FROM server_all_traffic_db WHERE sender = '{senderName}' ORDER BY relocation_timestamp'''
        result = mycursor.execute(create_view_query)
        
        
        connection.commit()
        connection.close()

    @staticmethod
    def senders():
        connection = sqlite3.connect('sqlite_model.db')
        mycursor = connection.cursor()

        query = f"SELECT DISTINCT sender FROM server_all_traffic_db"
        mycursor.execute(query)

        allSenders = mycursor.fetchall()
        senderNameList = [senderNameTuple[0] for senderNameTuple in allSenders]

        for senderName in senderNameList:
            viewName = f"{senderName}_SenderPerspective"
            CreateUsersView.createSendersView(viewName, senderName)

        connection.close()

    def generateviews():
      CreateUsersView().senders()
      CreateUsersView().receivers()
      