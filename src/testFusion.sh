#!/usr/bin/env bash
NEPOCHS=400
ALPHA=0.004
DELTA=20
python3 project.py fusion pantene $NEPOCHS $ALPHA $DELTA > Results/fusionPanteneResult.txt
python3 project.py fusion tampax $NEPOCHS $ALPHA $DELTA > Results/fusionTampaxResult.txt
python3 project.py fusion oral-b $NEPOCHS $ALPHA $DELTA > Results/fusionOralbResult.txt
python3 project.py fusion always $NEPOCHS $ALPHA $DELTA > Results/fusionAlwaysResult.txt
python3 project.py fusion gillette $NEPOCHS $ALPHA $DELTA > Results/fusionGilletteResult.txt
