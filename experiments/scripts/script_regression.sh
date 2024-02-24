#!/bin/bash

# If you want to perform GP regression without the python wrapper of RGaSP, add the option -out=x  (except for wwl)
# Where x = 3 if dataset_name is "Rotor37" or "Rotor37_CM"
#       x = 0 if dataset_name is "Tensile2d" or "Tensile2d_CM"
#       x = 0 if dataset_name is "AirfRANS" or "AirfRANS_CM"

conda activate ENV
cd ..

#swwl 1 all datasets
#for dataset in "Rotor37" "Rotor37_CM" "Tensile2d" "Tensile2d_CM" "AirfRANS" "AirfRANS_CM";
#do python run_regression.py --dataset=$dataset --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=1 --seeds 0 1 2 3 4 --verbose=1;
#done;

#swwl \sqrt(n) all datasets
# python run_regression.py --dataset=Rotor37 --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=173 --seeds 0 1 2 3 4 --verbose=1;
#python run_regression.py --dataset=Rotor37_CM --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=30 --seeds 0 1 2 3 4 --verbose=1;
# python run_regression.py --dataset=Tensile2d --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=100 --seeds 0 1 2 3 4 --verbose=1;
#python run_regression.py --dataset=Tensile2d_CM --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=30 --seeds 0 1 2 3 4 --verbose=1;
# python run_regression.py --dataset=AirfRANS --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=100 --seeds 0 1 2 3 4 --verbose=1;
# python run_regression.py --dataset=AirfRANS_CM --kernel=swwl --regressor=rgasp -H 3 -Q=500 -P=50 -T=100 --seeds 0 1 2 3 4 --verbose=1;

# propag just CM
#for tmax in 1; # 1 3 5 7
#do for w in 0.01; # 0.00001 0.0001 0.001 0.01 0.1"
#do python run_regression.py --dataset=Rotor37_CM --kernel=propag --regressor=rgasp -pk_tmax $tmax -pk_w $w --seeds 0 1 2 3 4 --verbose=1;
#do python run_regression.py --dataset=Tensile2d_CM --kernel=propag --regressor=rgasp -pk_tmax $tmax -pk_w $w --seeds 0 1 2 3 4 --verbose=1;
# do python run_regression.py --dataset=AirfRANS_CM --kernel=propag --regressor=rgasp -pk_tmax $tmax -pk_w $w --seeds 0 1 2 3 4 --verbose=1;
#done;
#done;

# Impact of P, Q for Rotor37
# for P in 2 5 10 20 30 40 50; # 2 5 10 20 30 40 50
# do for Q in 1000; #10 100 500 1000;
# do python run_regression.py --dataset=Rotor37 --kernel=swwl --regressor=rgasp -H 3 -Q=$Q -P=$P -T=1 --seeds 0 1 2 3 4 --verbose=1;
# done;
# done;
