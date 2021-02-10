#!/bin/bash

parentdir="/mnt/raid/data/SFB1315/SCZ/rsMEG/Analysis/FunctCon"
gamma1old=$(find $parentdir -maxdepth 2 -name *34_FC_orth-corr.npy)
gamma2old=$(find $parentdir -maxdepth 2 -name *40_FC_orth-corr.npy)

for file in $gamma1old; do 
    newfile=${file%34_FC_orth-corr.npy}Low-Gamma-1_FC_orth-corr.npy
    cp $file $newfile
done

for file in $gamma2old; do 
    newfile=${file%40_FC_orth-corr.npy}Low-Gamma-2_FC_orth-corr.npy
    cp $file $newfile
done
