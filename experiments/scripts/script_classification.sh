#!/bin/bash

conda activate ENV
cd ..

for dataset in "Cuneiform"; # "BZR" "COX2" "ENZYMES" "PROTEINS" "Cuneiform"
    do python run_classification.py --dataset=$dataset --kernel=swwl --classifier=svc -H 0 1 2 3 -Q=20 -P=20 -T=1 --seeds 0 1 2 3 4 5 6 7 8 9 --verbose=1;
    #do python run_classification.py --dataset=$dataset --kernel=wwl --classifier=svc -H 0 1 2 3 -T=1 --seeds 0 1 2 3 4 5 6 7 8 9 --verbose=1;
    #do python run_classification.py --dataset=$dataset --kernel=fgw --classifier=svc_rbf -alphas 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2 3 4 5 6 7 8 9 --verbose=1;
    #do python run_classification.py --dataset=$dataset --kernel=fgw --classifier=svc -alphas 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2 3 4 5 6 7 8 9 --verbose=1;
    #do python run_classification.py --dataset=$dataset --kernel=propag -pk_tmax 1 3 5 7 -pk_w 0.00001 0.0001 0.001 0.01 0.1 --seeds 0 1 2 3 4 5 6 7 8 9 --verbose=1;
    #do python run_classification.py --dataset=$dataset --kernel=ghopper --seeds 0 1 2 3 4 5 6 7 8 9 --verbose=1;
done;

# The sgml kernel requires tensorflow.
# conda activate ENV_WITH_TF
# for dataset in "Cuneiform"; # "BZR" "COX2" "ENZYMES" "PROTEINS" "Cuneiform"
#     do python run_classification.py --dataset=$dataset --kernel=sgml --classifier=svc -H 1 2 3 --seeds 0 1 2 3 4 5 6 7 8 9 --verbose=1;
# done;
