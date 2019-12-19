'''
Sources:
Hands-on Machine Learning with Scikit-Learn, Keras, and Tensorflow, 2nd Edition
by Aurelien Geron Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9

'''

import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


#first we will define the url which contains raw data and then use os.path to join datasets and housing to make datasets/housing
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

#this function creates a path called datasets/housing/ to put the dataset in
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
	os.makedirs(housing_path, exist_ok=True)
	tgz_path = os.path.join(housing_path, "housing.tgz")
	urllib.request.urlretrieve(housing_url, tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()

def display_scores(scores):
	print("scores: ", scores)
	print("Mean: ", scores.mean())
	print("Standard Deviation: ", scores.std())

#this reads the csv file downloaded and stores it as a datasframe
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)



def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]



'''
np.random.permutation gives us random permutations and in this example it shuffles our data rows in our dataset
it will then take the test_ratio of elements from the shuffled indices and save the rest for test

This is not the best method because you will fetch the same dataset everytime. You can fix that by calling np.random.seed(n) however that will cause another issue because we will eventually train with our test data.
Example:
>>> np.random.permutation(10)
array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
'''
def split_train_test(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]





#fetch_housing_data()
housing = load_housing_data()


#see some info about the dataset we are using
#print(housing.head())
#print(housing.info())

#print(housing["ocean_proximity"].value_counts())
#null values are not included in count
#print(housing.describe())

#housing.hist(bins=50, figsize=(20,15))
#plt.show()

#train_set, test_set = split_train_test(housing, 0.2)
#print(len(train_set))
#print(len(test_set))


#this method can be used but wont be representative of the data
train_set, test_set = train_test_split(housing, test_size=.2, random_state=42)

'''
it is very important to have enough instances in the dataset for each stratum so that it is representative of your data.

For example:
population is 51% women and 49% men. If you take a random sample and the sample contains 30% women and 70% men this would not be representative
'''

housing["income_categories"] = pd.cut(housing["median_income"],
	bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
	labels=[1, 2, 3, 4, 5])

#histogram to see income placed in categories
#housing["income_categories"].hist()
#plt.show()


#this will split the data and give us representative data using income categories
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_categories"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]


#this will give us representative data we saw on the histogram
print(strat_test_set["income_categories"].value_counts() / len(strat_test_set))


#now lets remove income_categories so the data will be back to what we had it as
for set_ in (strat_train_set, strat_test_set):
	set_.drop("income_categories", axis=1, inplace=True)

'''
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()
'''
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#we can also see the correlation matrix in a plot form
#attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
#scatter_matrix(housing[attributes], figsize=(12, 8))
#plt.show()

#from the plotting above we can see that only median_house_value and median_income are correlated, lets just plot that
#housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.show()

print(housing['median_income'])

'''
we can see that number of rooms is useless if we dont use the number of households with that number.
To make the number of rooms useful, we will have to divide number of rooms by total households to get
the number of rooms per household

'''

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

#check the correlation matrix again

corr_matrix = housing.corr()
print("Correlation matrix for Median house value with added derived fields: rooms_per_household, bedrooms_per_room, population_per_household")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

'''
from the info we achieved earlier from housing.info() we found that total_bedrooms had missing values.
We have three options to deal with this:
1. Get rid of the corresponding districts: housing.dropna(subset=["total_bedrooms"])
2. Get rid of the whole attribute: housing.drop("total_bedrooms", axis=1)
3.Set the values to some value (zero, mean, median) median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)
'''

#sklearn has a class to handle missing values

imputer = SimpleImputer(strategy="median")

#since median can only be calculated on numerical values we need to make a copy of the data without text attributes
# we will need to drop the text attribute ocean_proximity

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)


'''
to view the values we created from the function above we can print either of the below
imputer.statistics_
housing_num.median()values
'''

#now we can use the trained imputer to transform the training set which gives us a numpy array
X = imputer.transform(housing_num)

#then we can put it into a pandas df
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

#lets now address our problem with the ocean_proximity text values
#since there are a limited number of possible values for ocean_proximity we can encode them
housing_categories = housing[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_categories)
#print(housing["ocean_proximity"].value_counts())

#print(ordinal_encoder.categories_)

'''
The problem with using this encoder is that most ML algorithms will treat close values as more similar and in our case
that doesnt make much sense. In our case the ocean proximity have categorical values.
If however the ocean proximity was measured in miles away from the ocean, it would make more sense to using normal miles rather
than one hot encoding
'''

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_categories)

'''
one hot encoder does not store the whole matrix into memory. Instead it stores the location
of the one in each row so that the whole matrix filled with zeros wont be wasted by being putted into memory
'''

#to view the matrix how it will be interpreted we can print out the below code
#print(housing_cat_1hot.toarray())

'''
Feature Scaling
very important for machine learning algorithms. If one field has a larger scale than another, the algorithm will become biased
min-max scaling is when values are shifted and rescaled to end up ranging from 0 to 1. (x-min)/(max-min)

standardization is better because unlike min-max scaling, outliers will not drastically change the dataset.
always has a mean of zero. (x-mean)/(standard-deviation)
no specific range
'''

'''
Transformation Pipelines
scikit allows for a sequence of transformations by using pipelines
'''


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)


#this will be a standard pipeline for the numerical fields, all except ocean_proximity
#Standardscaler is standardization
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

#print(housing_num_tr)

#housing_num has all the values for numerical attributes. All except: ocean_proximity

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


#pipeline for our numerical values and for non numerical values which will use one hot encoding
'''
num_attribs give us all the numerical attributes in a list form
cat_attribs gives us all the categorical attributes
the full pipeline will perform all the transformations we defined in num_pipeline to each column in num_attribs
cat_attribs uses the onehotencoder because of categorical values
'''
full_pipeline = ColumnTransformer([
	("num", num_pipeline, num_attribs),
	("cat", OneHotEncoder(), cat_attribs)])

housing_prepared = full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("linear_model root mean squared error: ", lin_rmse)

'''
This model could have one or more of these problems:
1. Get a more powerful model
2. feed the algorithm with better features
3. Reduce contraints on the model

Lets use a decision tree regressor and see if we can get better results
'''

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


#an example of overfitting
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Tree_regressor root mean squared error: ", tree_rmse)

'''
To get a more reliable score of how the above model did we can use k-fold-cross-validation
here is an excerpt from this website to better help explain
https://machinelearningmastery.com/k-fold-cross-validation/
given dataset
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
split into k=3 folds
Fold1: [0.5, 0.2]
Fold2: [0.1, 0.3]
Fold3: [0.4, 0.6]
Model1: Trained on Fold1 + Fold2, Tested on Fold3
Model2: Trained on Fold2 + Fold3, Tested on Fold1
Model3: Trained on Fold1 + Fold3, Tested on Fold2
'''

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)


'''
GridSearch
Suppose you want to find some of the best hyperparamters to use and you dont have the time to fiddle with these combinations
you can use scikits GridSearchCV. The following code will run a RandomForestRegressor with different hyperparamters.
The first set will run 3*4=12 combinations to find the best result
The second set will run 3*2=6 combinations to find the best result
'''
param_grid = [{'n_estimators': [3,10, 30], 'max_features': [2, 4, 6, 8]},{'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2, 3, 4]}]
forest_reg = RandomForestRegressor()

print(forest_reg.get_params().keys())
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

print("best params:")
print(grid_search.best_params_)

#get best estimator
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score), params)


'''
If we want to see the importance of different features
'''
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))

'''
Now to test for the final model
'''

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)



