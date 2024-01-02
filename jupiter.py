import sqlite3
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class Moons:
	def __init__(self, db):

		database_service = "sqlite"
		self.connectable = f"{database_service}:///{db}"
		self.load_data() #creates the Moons class which will extract the data from the jupiter.db database
    
	def load_data(self):
       
		query = "SELECT * FROM moons"
		self.data = pd.read_sql(query, self.connectable) #creates an sql for the database data

	def stats(self, decimal_places=2):

		statdp = self.data.describe()
		statdp2 = statdp.round(decimal_places)
		return statdp2 #calculates the statistics of all of the variables to 2 decimal places

	def correlations(self, decimal_places=2):

		corrdp = self.data.corr()
		corrdp2 = corrdp.round(decimal_places)
		return corrdp2 #calculates the correlation coefficients between variables to 2 decimal places

	def plot_hist(self):
		plt.hist(self.data['distance_km'], bins=20, color='skyblue', edgecolor='black')
		plt.title('Distribution of Moon Distances from Jupiter')
		plt.xlabel('Distance (km)')
		plt.ylabel('Frequency')
		plt.show() #plots a histogram of the distance of the moons from jupiter


	def group_plot(self):
		groups = self.data['group'].unique()
		for group in groups:
			group_data = self.data[self.data['group'] == group]
			plt.scatter(group_data['period_days'], group_data['distance_km'], label=group) #isolates all of the seperate groups of moons so each can be individually seen
		plt.xlabel('Period Days')
		plt.ylabel('Distance km')
		plt.legend(title='Groups', loc='lower right')
		plt.title('Scatter Plot of Period Days vs Distance km by Group')
		plt.show() #plots a scatter graph of the distance of the moons to jupiter and the period days they have
		sns.displot(data= self.data, x="distance_km", hue="group", col="group")
		plt.show() #creastes individual histograms for eac type of moon

	def moon(self, moon_name):
        
		moon_data = self.data[self.data['moon'] == moon_name]

		return moon_data #allows individual moons to be searched for

	def linear_regression_model(self):
               
		self.data['T_seconds'] = self.data['period_days'] * 24 * 60 * 60 #converts period days into period seconds, the variable T
		self.data['T2'] = self.data['T_seconds'] **2 #squares T to produce T^2
		self.data['a_m'] = self.data['distance_km'] * 1000 #turns the variable a to metres from kilometres
		self.data['a3'] = self.data['a_m'] **3 #cubes the variable a to make a^3
		Y = self.data['T2'].values
		X = self.data['a3'].values.reshape(-1, 1) #reshaping to make a 2D array which is necessary
        
		sns.regplot(data=self.data, y='T2', x='a3', scatter_kws={'s': 15}, line_kws = {'linewidth': 1}) #plots a scatter graph of T^2 vs a^3, adjusting the size of the plots and line of best fit
		plt.title("Scatter Plot")
		plt.show()

		sns.residplot(data=self.data, y='T2', x='a3') #creates a residual plot based off the scatter above
		plt.title("Residual Plot")
		plt.show()
        

		# separate data into training and testing sets
		# use 30% of the data for testing, and the rest for training
		from sklearn.model_selection import train_test_split
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

		model = linear_model.LinearRegression(fit_intercept= False) #hyperparameter set here as I made fit_intercept= False, this is so the line of best fit passes through the origin, as a3 and t2 are proportional as well as keplers third law equation having no c value

		model.fit(x_train,y_train)
        
		pred = model.predict(x_test) #creates a prediction of the x_test

		f, a0 = plt.subplots(figsize=(15, 15))
		a0.scatter(self.data["a3"], self.data["T2"], label="Actual") #plots a scatter of t2 vs a3, 'actual' values
		a0.plot(x_test.flatten(), pred.flatten(), 'r--', label="Predicted") #plots the line of best fit using the 'prediction' values

		plt.title("Scatter Plot with Predicted Model")
		plt.legend()
		a0.set_xlabel("a3")
		a0.set_ylabel("T2")
		plt.show()
        
		fig, ax = plt.subplots()

		# Create a plot of residuals
		ax.plot(x_test,y_test - pred,'.')
		
		# Add a horizontal line at zero to guide the eye
		ax.axhline(0, color='k', linestyle='dashed')
	
		# Add axis labels
		ax.set_xlabel("T2")
		ax.set_ylabel("Residuals")
		ax.set_title("Residual Plot with Predicted Model")
		plt.show()

        
		from sklearn.metrics import r2_score, mean_squared_error
		print(f"unweighted model r2_score: {r2_score(y_test,pred)}") #calculates the r2 value to show the correlations
		print(f"unweighted model root mean squared error: {mean_squared_error(y_test,pred)}") #calculates root mean squared error
        
		gradient = model.coef_[0] #works out gradient of line of best fit
        
		print("gradient from model: ", gradient)
		print("intercept from model:", model.intercept_) #works out y-intercept
		print(f"root mean squared error: {mean_squared_error(y_test,pred, squared=False)}")
	

		print(f"4π/2GM is = {gradient}") #applying the equation to the graph 
		G = 6.67e-11 #m3kg−1s−2
		pi = np.pi #pi value imported
		M = (4*(pi)) / (2*(G)*(gradient)) #works out Mass of Jupiter in kg importing all other values into the equation
		print(f"The predicted mass of jupiter from my model is: {M}kg")
		actm =  1.899e+27 #kg (actual mass of Jupiter)
		print("The actual mass of jupiter is  1.899e+27 kg")
		print(f"So the actual mass of jupiter is {((actm)/(M))} times heavier than my estimation") #compares estimation to literature value
	
