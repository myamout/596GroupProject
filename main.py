import pandas as pd
import sqlite3
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
  connection = sqlite3.connect("database.sqlite")
  cursor = connection.cursor()

  # grab all of the players
  cursor.execute("SELECT * FROM Player")
  players_dataframe = pd.DataFrame(cursor.fetchall(), 
    columns=["id", "player_api_id", "player_name", "player_fifa_api_id", "birthday", "height", "weight"])

  # grab all of the players' attributes
  cursor.execute("SELECT * FROM Player_Attributes")
  players_attributes_dataframe = pd.DataFrame(cursor.fetchall(), 
    columns=["id", "player_fifa_api_id", "player_api_id", "date", "overall_rating", "potential", "preferred_foot", "attacking_work_rate", "defensive_work_rate", "crossing", "finishing", "heading_accuracy", "short_passing", "volleys", "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control", "acceleration", "sprint_speed", "agility", "reactions", "balance", "shot_power", "jumping", "stamina", "strength", "long_shots", "aggression", "interceptions", "positioning", "vision", "penalties", "marking", "standing_tackle", "sliding_tackle", "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"])
  

if __name__ == '__main__':
  main()
