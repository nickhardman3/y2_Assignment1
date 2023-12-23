import pandas as pd
import matplotlib.pyplot as plt

class Moons:
    def __init__(self, db):
    
        database_service = "sqlite"
        self.connectable = f"{database_service}:///{db}"
        self.load_data()
    
    def sql_table(self):
       
        query = "SELECT * FROM moons"
        self.data = pd.read_sql(query, self.connectable)

    def stats(self):

	return self.data.describe()
