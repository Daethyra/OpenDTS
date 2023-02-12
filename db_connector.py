import logging
import traceback
import pandas as pd
import psycopg2
import psycopg2.pool
import os
import dotenv
import re
import tempfile
import shutil
import zipfile
from functools import lru_cache
from datetime import datetime, timedelta

dotenv.load_dotenv()

ARCHIVE_FOLDER = 'archive'

def connect_to_db():
    try:
        # Create a connection pool
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, # minimum number of connections in the pool
            20, # maximum number of connections in the pool
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT")
        )
        # Get a connection from the pool
        connection = connection_pool.getconn()

        # Set autocommit to True for executemany to work
        connection.autocommit = True

        # Check if the archive folder exists
        if not os.path.exists(ARCHIVE_FOLDER):
            os.makedirs(ARCHIVE_FOLDER)
            first_run = True
        else:
            # Check the creation time of the archive folder
            archive_creation_time = datetime.fromtimestamp(os.path.getctime(ARCHIVE_FOLDER))
            threshold_time = datetime.now() - timedelta(days=28)
            first_run = False

            # If the archive folder was created more than 28 days ago, archive log files and any other files meeting the threshold
            if archive_creation_time < threshold_time:
                # Create a zip file of the log files and any other files meeting the threshold
                log_files = [f for f in os.listdir() if f.endswith('.log') and os.stat(f).st_ctime < threshold_time.timestamp()]
                files_to_archive = log_files # add other files to archive here
                archive_filename = f'{ARCHIVE_FOLDER}/archive_{datetime.now().strftime("%H-%M_%Y-%m-%d")}.zip'
                with zipfile.ZipFile(archive_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zip:
                    for file in files_to_archive:
                        zip.write(file)

                # Remove the original log files and any other files that were archived
                for file in files_to_archive:
                    os.remove(file)

                # Update the creation time of the archive folder
                os.utime(ARCHIVE_FOLDER, None)

        # Return the connection with the added executemany functionality and first run flag
        return connection, connection.cursor().execute if not hasattr(connection.cursor(), 'executemany') else connection.cursor().executemany, first_run
    except Exception as e:
        # Log error
        logging.error(f"An error occurred while connecting to the database: {e}")
        logging.error(traceback.format_exc())
        raise e
