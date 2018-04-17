import pandas as pd
import sqlite3
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def main():
  connection = sqlite3.connect("database.sqlite")
  cursor = connection.cursor()

  # grab all of the players
  cursor.execute("SELECT * FROM Player")
  players = pd.DataFrame(cursor.fetchall(), 
    columns=["id", "player_api_id", "player_name", "player_fifa_api_id", "birthday", "height", "weight"])

  # grab all of the players' attributes
  cursor.execute("SELECT * FROM Player_Attributes")
  players_attributes = pd.DataFrame(cursor.fetchall(), 
    columns=["id", "player_fifa_api_id", "player_api_id", "date", "overall_rating", "potential", "preferred_foot", "attacking_work_rate", "defensive_work_rate", "crossing", "finishing", "heading_accuracy", "short_passing", "volleys", "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control", "acceleration", "sprint_speed", "agility", "reactions", "balance", "shot_power", "jumping", "stamina", "strength", "long_shots", "aggression", "interceptions", "positioning", "vision", "penalties", "marking", "standing_tackle", "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"])
  
  # I'm not sure we even need the players_dataframe
  # we'll see

  # replace attacking work rate and defensive work rate with numbers
  # 1 = low, 2 = medium, 3 = high
  # These steps are kinda repeatative, we can fix this
  players_attributes.attacking_work_rate.replace(["low", "medium", "high"], [1, 2, 3], inplace=True)
  players_attributes.defensive_work_rate.replace(["low", "medium", "high"], [1, 2, 3], inplace=True)

  # Remove any rows with NaN
  players_attributes = players_attributes.dropna()

  # Labels (potential)
  labels = players_attributes["potential"].values
  
  # Drop shit we don't need
  players_attributes.drop("overall_rating", axis=1, inplace=True)
  players_attributes.drop("id", axis=1, inplace=True)
  players_attributes.drop("player_fifa_api_id", axis=1, inplace=True)
  players_attributes.drop("player_api_id", axis=1, inplace=True)
  players_attributes.drop("date", axis=1, inplace=True)
  players_attributes.drop("potential", axis=1, inplace=True)
  players_attributes.drop("preferred_foot", axis=1, inplace=True)
  players_attributes.drop("attacking_work_rate", axis=1, inplace=True)
  players_attributes.drop("defensive_work_rate", axis=1, inplace=True)

  # Prints out our row x column size
  print(players_attributes.shape)
  
  # Just the columns we want to use
  feature_columns = ["attacking_work_rate", "defensive_work_rate", "crossing", "finishing", "heading_accuracy", "short_passing", "volleys", "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control", "acceleration", "sprint_speed", "agility", "reactions", "balance", "shot_power", "jumping", "stamina", "strength", "long_shots", "aggression", "interceptions", "positioning", "vision", "penalties", "marking", "standing_tackle", "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
  # Remove any row that contains something other than an int or a float
  
  # Create our feature data

  features = players_attributes.iloc[:, :players_attributes.shape[1]-1]

  x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=5)

  lr = LinearRegression()
  model = lr.fit(x_train, y_train)
  print(model.score(x_test, y_test))
  # Print out more stats for the linear method
  # Try using Neural Network or some other model for better results?

  # Neural Networks
  mlr = MLPRegressor(hidden_layer_sizes=100, activation="tanh", solver="adam", learning_rate_init=0.001)
  fnn_model = mlr.fit(x_train, y_train)
  print(fnn_model.score(x_test, y_test))




if __name__ == '__main__':
  main()
