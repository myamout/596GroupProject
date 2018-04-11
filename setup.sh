#!bin/bash
python3 -m venv .
source ./bin/activate
pip install sklearn matplotlib numpy pandas scipy
unzip database.sqlite.zip
