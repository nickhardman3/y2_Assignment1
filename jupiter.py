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

    def stats(self, decimal_places=2):

	statdp = self.data.describe()
	statdp2 = statdp.round(decimal_places)
	return statdp2

    def correlations(self, decimal_places=2):

        corrdp = self.data.corr()
	corrdp2 = corrdp.round(decimal_places)
	return corrdp2

    def plot_hist(self):
        plt.hist(self.data['distance_km'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Moon Distances from Jupiter')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.show()


    def group_plot(self):
        groups = self.data['group'].unique()
        for group in groups:
            group_data = self.data[self.data['group'] == group]
            plt.scatter(group_data['period_days'], group_data['distance_km'], label=group)
        plt.xlabel('Period Days')
        plt.ylabel('Distance km')
        plt.legend(title='Groups', loc='lower right')
        plt.title('Scatter Plot of Period Days vs Distance km by Group')
        plt.show()

    def moon(self, moon_name):
        
        moon_data = self.data[self.data['moon'] == moon_name]

        return moon_data
