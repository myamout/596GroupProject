import pandas as pd
import sqlite3
import numpy as np
from sklearn.linear_model import LinearRegression
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

  # Labels (overall_rating)
  labels = players_attributes["overall_rating"].values
  
  # Drop shit we don't need
  players_attributes.drop("overall_rating", axis=1, inplace=True)
  players_attributes.drop("id", axis=1, inplace=True)
  players_attributes.drop("player_fifa_api_id", axis=1, inplace=True)
  players_attributes.drop("player_api_id", axis=1, inplace=True)
  players_attributes.drop("date", axis=1, inplace=True)
  players_attributes.drop("potential", axis=1, inplace=True)
  players_attributes.drop("preferred_foot", axis=1, inplace=True)

  print(players_attributes.shape)

  # We need to do more data clean up, some of the values are not ints or floats
  # THIS IS REALLY SLOW WE NEED TO FIX THIS!
  for index, row in players_attributes.iterrows():
    for key, value in row.items():
      if isinstance(value, float) == False and isinstance(value, int) == False:
        players_attributes.drop(players_attributes.index[index])
  
  # Create our feature data
  features = players_attributes.iloc[:, :players_attributes.shape[1]-1]

  x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=5)

  lr = LinearRegression()
  model = lr.fit(x_train, y_train)
  print(model.predict(x_test))



if __name__ == '__main__':
  main()
