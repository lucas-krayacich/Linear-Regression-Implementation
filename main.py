import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Ecommerce Customers.csv")
customers.head()
customers.describe()
print(customers.info())

sns.set_palette("GnBu_d")
sns.set_style("whitegrid")

#Basic Data Visualization

seshVSspend = sns.scatterplot(customers, x='Avg. Session Length', y='Yearly Amount Spent')

appVSspend = sns.scatterplot(customers, x='Time on App', y ='Yearly Amount Spent')

allAttributes = sns.pairplot(customers)

#Separation into Train and Test Sets

y = customers['Yearly Amount Spent']
x = customers[['Avg. Session Length','Time on App', 'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

#Fitting to Linear Regression Model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

#Coefficient -- single unit increase in x_attribute creates a 'coefficient' impact on the target

print('\nCoefficients: \n', lm.coef_)

coefficients = pd.DataFrame(lm.coef_, x.columns)
coefficients.columns = ['Coefficient']
print('\n', coefficients)
#coefficents measured in total dollars spent (target units)


predictions = lm.predict(x_test)

plt.scatter(y_test, predictions)

#Model Evalutation:

from sklearn import metrics

print('\n\nMAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))




