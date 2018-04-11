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
  players = cursor.fetchall()

  # grab all of the players' attributes
  cursor.execute("SELECT * FROM Player_Attributes")
  player_attributes = cursor.fetchall()

  print(players[1:10])
  print(player_attributes[1:0])

if __name__ == '__main__':
  main()
