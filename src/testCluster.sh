#!/usr/bin/env bash
NEPOCHS=400
ALPHA=0.004
DELTA=20
python3 project.py cluster pantene $NEPOCHS $ALPHA $DELTA > panteneResult.txt
python3 project.py cluster tampax $NEPOCHS $ALPHA $DELTA > tampaxResult.txt
python3 project.py cluster oral-b $NEPOCHS $ALPHA $DELTA > oralbResult.txt
python3 project.py cluster always $NEPOCHS $ALPHA $DELTA > alwaysResult.txt
python3 project.py cluster gillette $NEPOCHS $ALPHA $DELTA > gilletteResult.txt
