#!/usr/bin/env bash
MOD1=base
MOD2=cluster
MOD3=average
MOD4=fusion
PROD1=pantene
PROD2=tampax
PROD3=gillette
PROD4=always
PROD5=oral-b
NEPOCHS=400
ALPHA=0.004
DELTA=20
python3 project.py $MOD1 $MOD2 $PROD1 $NEPOCHS $ALPHA $DELTA
python3 project.py $MOD1 $MOD2 $PROD2 $NEPOCHS $ALPHA $DELTA
python3 project.py $MOD1 $MOD2 $PROD3 $NEPOCHS $ALPHA $DELTA
python3 project.py $MOD1 $MOD2 $PROD4 $NEPOCHS $ALPHA $DELTA
python3 project.py $MOD1 $MOD2 $PROD5 $NEPOCHS $ALPHA $DELTA
