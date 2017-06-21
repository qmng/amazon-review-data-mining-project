#!/usr/bin/env bash
NEPOCHS=400
ALPHA=0.004
python3 project.py average pantene $NEPOCHS $ALPHA > Results/avgPanteneResult.txt
python3 project.py average tampax $NEPOCHS $ALPHA > Results/avgTampaxResult.txt
python3 project.py average oral-b $NEPOCHS $ALPHA > Results/avgOralbResult.txt
python3 project.py average always $NEPOCHS $ALPHA > Results/avgAlwaysResult.txt
python3 project.py average gillette $NEPOCHS $ALPHA > Results/avgGilletteResult.txt
