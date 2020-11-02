#!/bin/bash

parentdir="/mnt/raid/data/SFB1315/SCZ/rsMEG/Analysis/FunctCon"
gamma1old=$(find $parentdir -maxdepth 2 -name *32_FC_orth-corr.npy)
gamma2old=$(find $parentdir -maxdepth 2 -name *36_FC_orth-corr.npy)
gamma3old=$(find $parentdir -maxdepth 2 -name *40_FC_orth-corr.npy)
gamma4old=$(find $parentdir -maxdepth 2 -name *44_FC_orth-corr.npy)

delete=$(find $parentdir -maxdepth 2 -name *Freq-44_FC_orth-corr.npyGamma-3_FC_orth-corr.npy)

for file in $delete; do
    echo $file
    rm $file
done

for file in $gamma1old; do 
    newfile=${file%32_FC_orth-corr.npy}Gamma-1_FC_orth-corr.npy
    cp $file $newfile
done

for file in $gamma2old; do 
    newfile=${file%36_FC_orth-corr.npy}Gamma-2_FC_orth-corr.npy
    cp $file $newfile
done

for file in $gamma3old; do 
    newfile=${file%40_FC_orth-corr.npy}Gamma-3_FC_orth-corr.npy
    cp $file $newfile
done

for file in $gamma4old; do 
    newfile=${file%44_FC_orth-corr.npy}Gamma-4_FC_orth-corr.npy
    cp $file $newfile
done