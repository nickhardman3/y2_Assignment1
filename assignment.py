import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


database_service = "sqlite"
database = "jupiter.db"
connectable = f"{database_service}:///{database}"

query="SELECT distance_km FROM moons"

pd.read_sql(query, connectable)
