#!/bin/bash

cd ..

##### Rotor37_CM
#R CMD BATCH "--args Rotor37_CM wwl 3 1" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_CM_wwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Rotor37_CM wwl 3 30 " run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_CM_wwl_$SLURM_JOB_ID.txt
R CMD BATCH "--args Rotor37_CM swwl 3 1 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_CM_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Rotor37_CM swwl 3 30 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_CM_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Rotor37_CM propag 1 0.001" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_CM_propag_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Rotor37_CM propag 1 0.001" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_CM_propag_$SLURM_JOB_ID.txt

#### Rotor37
#R CMD BATCH "--args Rotor37 swwl 3 1 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Rotor37 swwl 3 173 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37_swwl_$SLURM_JOB_ID.txt

##### Tensile2d_CM
#R CMD BATCH "--args Tensile2d_CM wwl 3 1" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_CM_wwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Tensile2d_CM wwl 3 30" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_CM_wwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Tensile2d_CM swwl 3 1 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_CM_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Tensile2d_CM swwl 3 30 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_CM_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Tensile2d_CM propag 1 0.01" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_CM_propag_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Tensile2d_CM propag 1 0.001" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_CM_propag_$SLURM_JOB_ID.txt

#### Tensile2d
#R CMD BATCH "--args Tensile2d swwl 3 1 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args Tensile2d swwl 3 500 50 500" run_regression.R ./scripts/logs/o/regression_R/res_Tensile2d_swwl_$SLURM_JOB_ID.txt

##### AirfRANS_CM
#R CMD BATCH "--args AirfRANS_CM wwl 3 1" run_regression.R ./scripts/logs/o/regression_R/res_AirfRAN_CM_wwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args AirfRANS_CM wwl 3 500" run_regression.R ./scripts/logs/o/regression_R/res_AirfRAN_CM_wwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args AirfRANS_CM swwl 3 1 50 500" run_regression.R ./scripts/logs/o/regression_R/res_AirfRAN_CM_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args AirfRANS_CM swwl 3 500 50 500" run_regression.R ./scripts/logs/o/regression_R/res_AirfRAN_CM_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args AirfRANS_CM propag 1 0.01" run_regression.R ./scripts/logs/o/regression_R/res_AirfRAN_CM_propag_$SLURM_JOB_ID.txt
#R CMD BATCH "--args AirfRANS_CM propag 1 0.001" run_regression.R ./scripts/logs/o/regression_R/res_AirfRAN_CM_propag_$SLURM_JOB_ID.txt

##### AirfRANS_CM
#R CMD BATCH "--args AirfRANS swwl 3 1 50 500" run_regression.R ./scripts/logs/o/regression_R/res_AirfRANS_swwl_$SLURM_JOB_ID.txt
#R CMD BATCH "--args AirfRANS swwl 3 500 50 500" run_regression.R ./scripts/logs/o/regression_R/res_AirfRANS_swwl_$SLURM_JOB_ID.txt

#### Variations of P, Q for Rotor37
# for P in 2 5 10 20 30 40 50; # 2 5 10 20 30 40 50
# do for Q in 10 10 100 500 1000; # 10 10 100 500 1000
# do R CMD BATCH "--args Rotor37 swwl 3 1 $P $Q" run_regression.R ./scripts/logs/o/regression_R/res_Rotor37swwl_$SLURM_JOB_ID.txt;
# done;
# done;


