# Python 3 - MAB 2025
import os
from sqlalchemy import create_engine

def create_connection():
    """Create a connection to PostgreSQL using SQLAlchemy"""
    # Database connection parameters
    username = 'mab-desk'
    password = os.getenv("POSTGRES_GDB_PASSWORD")
    host = 'localhost'
    port = '5432'
    database = 'gdb9'
    
    # Create connection string
    connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'
    
    # Create engine
    engine = create_engine(connection_string)
    return engine
