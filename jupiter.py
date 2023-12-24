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

    def correlations(self):

        return self.data.corr()

    def plot(self):
        
        plt.scatter(self.data['distance_km'], self.data['period_days'])
        plt.xlabel('Distance from Jupiter (km)')
        plt.ylabel('Orbital Period (days)')
        plt.title('Distance vs. Orbital Period for Jupiter\'s Moons')
        plt.show()

    def moon(self, moon_name):
        
        moon_data = self.data[self.data['moon'] == moon_name]

        return moon_data
