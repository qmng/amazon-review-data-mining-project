#!/usr/bin/env bash
NEPOCHS=400
ALPHA=0.004
python3 project.py base pantene $NEPOCHS $ALPHA > Results/basePanteneResult.txt
python3 project.py base tampax $NEPOCHS $ALPHA > Results/baseTampaxResult.txt
python3 project.py base oral-b $NEPOCHS $ALPHA > Results/baseOralbResult.txt
python3 project.py base always $NEPOCHS $ALPHA > Results/baseAlwaysResult.txt
python3 project.py base gillette $NEPOCHS $ALPHA > Results/baseGilletteResult.txt
