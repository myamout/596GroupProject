import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

def main():
  connection = sqlite3.connect("database.sqlite")
  cursor = connection.cursor()

  # grab all of the players' attributes from the sqlite3 database
  cursor.execute("SELECT * FROM Player_Attributes")
  players_attributes = pd.DataFrame(cursor.fetchall(),
    columns=["id", "player_fifa_api_id", "player_api_id", "date", "overall_rating", "potential", "preferred_foot", "attacking_work_rate", "defensive_work_rate", "crossing", "finishing", "heading_accuracy", "short_passing", "volleys", "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control", "acceleration", "sprint_speed", "agility", "reactions", "balance", "shot_power", "jumping", "stamina", "strength", "long_shots", "aggression", "interceptions", "positioning", "vision", "penalties", "marking", "standing_tackle", "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"])

  # Remove any rows with NaN
  players_attributes = players_attributes.dropna()

  # Labels (overall rating)
  labels = players_attributes["overall_rating"].values

  # create a list on the individual attributes that we aren't going to use as features
  drop_list = ["id", "player_fifa_api_id", "player_api_id", "date", "overall_rating", "preferred_foot", "attacking_work_rate", "defensive_work_rate", ]

  # Drop everything
  for attribute in drop_list:
    players_attributes.drop(attribute, axis=1, inplace=True)

  # Create our feature data
  features = players_attributes.iloc[:, :players_attributes.shape[0]-1]

  # Use Sklearn to create our training and testing datasets
  x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=5)

  ######################
  # Gradient Descent Model
  ALPHA = 0.19
  theta = np.zeros(34)
  MAX_ITER = 1500
  sc = MinMaxScaler() #Standardizing dataset
  sc.fit(x_train,x_test)
  x_train_std = sc.transform(x_train) #Transforming the dataset to fit the gradient descent
  x_test_std = sc.transform(x_test)
  xvalues = np.ones((59517,34))
  xvalues[:,1:34] = x_test_std[:,0:33]
  def gradientDescent(X, y, theta, alpha, numIterations):
    m = len(y)
    arrCost =[]
    transposedX = np.transpose(X)
    transposedTheta = np.transpose(theta)
    for iteration in range(0, numIterations):
        guess = np.dot(X,theta)
        residualError = guess - y
        gradient = np.dot(transposedX,residualError) / m
        change = [alpha * x for x in gradient]
        theta = np.subtract(theta, change)
        atmp = np.sum(residualError ** 2)/(2*m)
        arrCost.append(atmp)
    return theta, arrCost
  
  [theta, arrCost] = gradientDescent(x_test_std,y_test,theta,ALPHA,MAX_ITER)

  plt.plot(range(0,len(arrCost)),arrCost)
  plt.xlabel('iteration')
  plt.ylabel('cost')
  plt.title('Convergence Curve')
  plt.show()

  testXValues = np.ones((len(x_test_std), 34)) 
  testXValues[:, 1:3] = x_test_std[:, 0:2]
  tVal =  testXValues.dot(theta)
  
  tError = np.sqrt([x**2 for x in np.subtract(tVal, x_test_std[:, 2])])
  print('results: {} ({})'.format(round(np.mean(tError),2), round(np.std(tError),2)))
  #############################################################################
  #Neural Networks
  mlpr = MLPRegressor(hidden_layer_sizes=100, activation="relu", solver="adam", learning_rate_init=0.0001)
  fnn_model = mlpr.fit(x_train, y_train)
  print("FFN - MLPRegressor")
  print("Accuracy: ", fnn_model.score(x_test, y_test))
  y_pred = fnn_model.predict(x_test)
  error = mean_absolute_error(y_pred, y_test)
  print("Error: ",error)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.scatter(y_pred,y_test, s=10, c='r', marker="o", label='NN Prediction')
  ax1.plot(x_test, x_test, c='b')
  ax1.set_ylim(ymin=30)
  ax1.set_xlim(xmin=30)
  plt.title('Accuracy')
  plt.xlabel('Predictions')
  plt.ylabel('Actual Values')
  plt.show()

  ####################
  # K Nearest Neighbor Model
  kn = KNeighborsRegressor(n_neighbors=12, weights='distance', n_jobs=-1)
  kn_model = kn.fit(x_train, y_train)
  y_pred = kn_model.predict(x_test)

  mean_error = mean_absolute_error(y_test, y_pred)
  kn_score = r2_score(y_test, y_pred)

  print("K Nearest Neighbors Regression")
  print("Accuracy: {}".format(kn_score))
  print("Mean Error: {}".format(mean_error))

  ax = plt.subplot(111)
  plt.scatter(y_pred, y_test, c='r', marker='x')
  plt.plot(x_test, x_test, 'b', lw=1)
  plt.ylim(ymin=30)
  plt.xlim(xmin=30)
  plt.xlabel('Red: Y Predictions, Blue: X Tests')
  plt.ylabel('Red: Y Tests, Blue: X Tests')
  plt.show()



if __name__ == '__main__':
  main()
