#!/bin/bash
python preprocess.py
python lstm_character_based.py
python generate_output.py