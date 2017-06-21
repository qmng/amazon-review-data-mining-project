#!/usr/bin/env bash
NEPOCHS=400
ALPHA=0.004
DELTA=20
python3 project.py cluster pantene $NEPOCHS $ALPHA $DELTA > Results/clusterPanteneResult.txt
python3 project.py cluster tampax $NEPOCHS $ALPHA $DELTA > Results/clusterTampaxResult.txt
python3 project.py cluster oral-b $NEPOCHS $ALPHA $DELTA > Results/clusterOralbResult.txt
python3 project.py cluster always $NEPOCHS $ALPHA $DELTA > Results/clusterAlwaysResult.txt
python3 project.py cluster gillette $NEPOCHS $ALPHA $DELTA > Results/clusterGilletteResult.txt
