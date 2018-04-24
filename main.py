import pandas as pd
import tensorflow as tf
import tempfile as temp
import sqlite3
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# todo write function to get a batch of data

def get_data():
  connection = sqlite3.connect("database.sqlite")
  cursor = connection.cursor()

  # get the players
  cursor.execute("SELECT * FROM Player")
  players = pd.DataFrame(cursor.fetchall(), 
  columns= ["id", "player_api_id", "player_name", "player_fifa_api_id", "birthday", "height", "weight"])

  # get player attributes
  cursor.execute("SELECT * FROM Player_Attributes")
  players_attributes = pd.DataFrame(cursor.fetchall(), 
  columns=["id", "player_fifa_api_id", "player_api_id", "date", "overall_rating", "potential", "preferred_foot", "attacking_work_rate", "defensive_work_rate", "crossing", "finishing", "heading_accuracy", "short_passing", "volleys", "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control", "acceleration", "sprint_speed", "agility", "reactions", "balance", "shot_power", "jumping", "stamina", "strength", "long_shots", "aggression", "interceptions", "positioning", "vision", "penalties", "marking", "standing_tackle", "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"])

  # replace attacking work rate and defensive work rate with numbers
  # 1 = low, 2 = medium, 3 = high
  # These steps are kinda repeatative, we can fix this
  players_attributes.attacking_work_rate.replace(["low", "medium", "high"], [1, 2, 3], inplace=True)
  players_attributes.defensive_work_rate.replace(["low", "medium", "high"], [1, 2, 3], inplace=True)

  # Remove any rows with NaN
  players_attributes = players_attributes.dropna()

  # Labels (potential)
  labels = np.array(players_attributes["potential"].values)

  # Drop data we don't need
  # data used for predicting overall, for some reason positioning is not showing up
  # ball_control, dribbling, "sprint_speed", "agility", "acceleration", "short_passing", "long_passing", "interceptions", "stamina", "balance", "volleys", "strength", "crossing", "reactions", "jumping", "vision", "aggression", "penalties", "marking",
  unwanted_columns=["id", "player_fifa_api_id", "player_api_id", "positioning",  "date", "overall_rating", "potential", "preferred_foot", "attacking_work_rate", "defensive_work_rate", "finishing", "heading_accuracy", "curve", "free_kick_accuracy", "shot_power", "long_shots", "standing_tackle", "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
  for column in unwanted_columns:
    players_attributes.drop(column, axis=1, inplace=True)
  
  # Create our feature data
  features = players_attributes.iloc[:, :players_attributes.shape[1]-1]
  features = feature_normalize(features)
  features = np.array(features)

  f, l = append_bias_reshape(features, labels)

  x_train, x_test, y_train, y_test = train_test_split(f, l, test_size=0.33, random_state=5)

  return x_train, x_test, y_train, y_test

def feature_normalize(dataset):
  mu = np.mean(dataset,axis=0)
  sigma = np.std(dataset,axis=0)
  return (dataset - mu)/sigma

def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l

def main():
  # Parameters
  learning_rate = 0.01
  training_epochs = 1000
  cost_history = np.empty(shape=[1], dtype=float)

  x_train, x_test, y_train, y_test = get_data()
  n_dim = x_train.shape[1]

  X = tf.placeholder(tf.float32,[None, n_dim])
  Y = tf.placeholder(tf.float32,[None, 1])
  W = tf.Variable(tf.ones([n_dim, 1]))

  init = tf.global_variables_initializer()

  y_ = tf.matmul(X, W)
  cost = tf.reduce_mean(tf.square(y_ - Y))
  training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  sess = tf.Session()
  sess.run(init)

  for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={X: x_train,Y: y_train})
    cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: x_train, Y: y_train}))

  plt.plot(range(len(cost_history)),cost_history)
  plt.axis([0, training_epochs, 0, np.max(cost_history)])
  plt.show()

  pred_y = sess.run(y_, feed_dict={X: x_test})
  mse = tf.reduce_mean(tf.square(pred_y - y_test))
  print("MSE: %.4f" % sess.run(mse)) 

  fig, ax = plt.subplots()
  ax.scatter(y_test, pred_y)
  ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')
  plt.show()

  sess.close()

if __name__ == '__main__':
  main()
