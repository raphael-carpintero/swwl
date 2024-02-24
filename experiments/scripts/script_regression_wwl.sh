#!/bin/bash

conda activate ENV
cd ..

# Step 1
python wwl_meshes_parallelized_step1.py --dataset Rotor37_CM -H=3 -T=1
#python wwl_meshes_parallelized_step1.py --dataset Tensile2d_CM -H=3 -T=1
#python wwl_meshes_parallelized_step1.py --dataset AirfRANS_CM -H=3 -T=1

# Step 2
# for line in 0;  #1 2 3 4... 99
# do python run_wl_no_parallelized2.py --dataset Rotor37 -H=3 -T=1 --line $line;
# done;

# Step 3
#python wwl_meshes_parallelized_step3.py --dataset Rotor37_CM -H=3 -T=1
#python wwl_meshes_parallelized_step3.py --dataset Tensile2d_CM -H=3 -T=1
#python wwl_meshes_parallelized_step3.py --dataset AirfRANS_CM -H=3 -T=1