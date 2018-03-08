#Recurrent Neural Network

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Part 1 - Data Preprocessing


#Importing Training set.
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#Creates the dataframe with "Open" coloumn
training_set = dataset_train.iloc[:,1:2].values


#Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
#It is going to fit the training_set and transforms the data.
scaled_training_set = sc.fit_transform(training_set)

#Creating the data structure with 60 timesteps and 1 output 
#In each time 't' , RNN will look 60 timesteps (days) before steps to predict the next output
X_train = []
y_train = []
for i in range(60, 1257):
    X_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
 
#Reshaping for giving X_train to the RNN as an input.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part - 2 Building the RNN

#Importing the KEras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initialising the RNN
#Represents sequence of layers : regressor(Because we will predict continious output, so we are doing regression not a classification.)
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation.
    #return_sequences : We set that True , because we want to implement stack of LSTM layers. 
    #If we want to add more LSTM layers afterr the current layer , we want to set this value True, if not we set that value to be False.
regressor.add(LSTM(units= 50, return_sequences= True, input_shape = ( X_train.shape[1], 1)))
#We are adding a Dropout Regularisation
regressor.add(Dropout(0.2))

#Adding a second LSTM layer and some Dropout Regularisation
#We are not spesifying the input shape, because it will understand by the previous one.
regressor.add(LSTM(units= 50, return_sequences= True))
#We are adding a Dropout Regularisation
regressor.add(Dropout(0.2))

#Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True))
#We are adding a Dropout Regularisation
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer and some Dropout regularisation.
#We are not going to return any sequencies.
regressor.add(LSTM(units= 50, return_sequences= False))
#We are adding a Dropout Regularisation.
regressor.add(Dropout(0.2))

#Adding the output layer.
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size= 32, epochs= 100)

 
# Part - 3 - Making the predictions and visualising the results.
    #Getting the real stock price of 2017
    #Importing Training set.
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    #Creates the dataframe with "Open" coloumn
real_stock_prices = dataset_test.iloc[:,1:2].values

#Getting the predicted stock prices of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
#Getting the first data row that we will use for the predict the first test data.
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#We are modifying the numpy array as all the inputs in one coloumn
inputs = inputs.reshape(-1,1)
#Scaling the input with fitted scaler
inputs = sc.transform(inputs)


X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
#Reshaping for giving X_test to the RNN as an input.
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Predicting the scaled stock prices
predicted_stock_price = regressor.predict(X_test)
#Inverse scaling and transforming
predicted_stock_price = sc.inverse_transform(predicted_stock_price) 

#Visualising the results
plt.plot(real_stock_prices, color= 'b', label= 'Real Stock Price of Google Jan 2017')
plt.plot(predicted_stock_price, color= 'g', label= 'Predicted Stock Price of Google Jan 2017')
plt.title('Google Stock Price Prediction with Recurrent Neural Networks')
plt.xlabel('Time')
plt.ylabel('Open Stock Prices')
plt.legend()
plt.show()