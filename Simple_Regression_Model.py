'''
Sources:
Hands-on Machine Learning with Scikit-Learn, Keras, and Tensorflow, 2nd Edition
by Aurelien Geron Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9

These datasets include GDP per capita and life satisfaction. Life satisfaction is rated from 1-10, 10 being the most satisfactory
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

def select_model(model_type):
	models={
		"linear": sklearn.linear_model.LinearRegression(),
		"knn": sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
	}
	return models[model_type]

#loading of the data. Here is where you can download the data http://www.edwardhk.com/language/python/hands-on-machine-learning-example-1-1/
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

#np.c_ is to concatenate. we are going to define our input and output (x and y) as GDP_per. 
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#plot the data
country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
plt.ylim(0,10)
plt.show()

# select the model linear or knn
model = select_model("linear")

#fit the model and show the coefficient m and intercept in the equation y=mx+b where x is the predictor and b is the constant or intercept
model.fit(x, y)
coefficient = model.coef_
intercept = model.intercept_
print "Equation of line: y=%fx+%.2f" % (coefficient, intercept)

# make a prediction for cyprus using Cyprus's GDP per capita
cyprus_GDP = [[22578]]
prediction = model.predict(cyprus_GDP)
print "Prediction of cyprus's life Satisfaction: %.2f" % (prediction)



