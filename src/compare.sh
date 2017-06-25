#!/usr/bin/env bash
MOD1=base
MOD2=cluster
MOD3=average
MOD4=fusion
PROD=always
NEPOCHS=400
ALPHA=0.004
DELTA=20
python3 project.py $MOD1 $MOD2 $PROD $NEPOCHS $ALPHA $DELTA > Results/$PROD/$MOD1-$MOD2-$NEPOCHS-$ALPHA-$DELTA.txt
