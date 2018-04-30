#!/bin/bash
mkdir -p submissions
for D in `find exercises/final -maxdepth 1 -mindepth 1 -type d`; do
    cd $D
    zip=B$(basename $(pwd))_ML_18_NguReichel_Blatt.zip
    zip -r $zip . && cd -
    mv $D/$zip submissions/
done
