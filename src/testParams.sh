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
for nepochs in `seq 100 100 1000`;
do
for n in `seq 0 -1 -4`;
do
alpha=$(awk "BEGIN{print 10 ** $n}")
for delta in `seq 10 10 100`;
do
python3 project.py $MOD1 $MOD2 $PROD1 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD2 $PROD2 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD2 $PROD3 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD2 $PROD4 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD2 $PROD5 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD3 $PROD1 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD3 $PROD2 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD3 $PROD3 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD3 $PROD4 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD3 $PROD5 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD4 $PROD1 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD4 $PROD2 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD4 $PROD3 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD4 $PROD4 $nepochs $alpha $delta
python3 project.py $MOD1 $MOD4 $PROD5 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD3 $PROD1 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD3 $PROD2 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD3 $PROD3 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD3 $PROD4 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD3 $PROD5 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD4 $PROD1 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD4 $PROD2 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD4 $PROD3 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD4 $PROD4 $nepochs $alpha $delta
python3 project.py $MOD2 $MOD4 $PROD5 $nepochs $alpha $delta
python3 project.py $MOD3 $MOD4 $PROD1 $nepochs $alpha $delta
python3 project.py $MOD3 $MOD4 $PROD2 $nepochs $alpha $delta
python3 project.py $MOD3 $MOD4 $PROD3 $nepochs $alpha $delta
python3 project.py $MOD3 $MOD4 $PROD4 $nepochs $alpha $delta
python3 project.py $MOD3 $MOD4 $PROD5 $nepochs $alpha $delta
done
done
done
