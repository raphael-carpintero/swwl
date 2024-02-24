# !/usr/bin/env Rscript

# This file is used to use Gaussian process regression with precomputed distance/Gram matrices. It should be called after running run_{classification, regression}.py.
# *** Usage *** R CMD BATCH "--args Rotor37_CM swwl 0 1 50 500" run_regression.R run_regression_out.txt

args = commandArgs(trailingOnly=TRUE)

dataset_name <- args[1]
kernel <- args[2]
if (kernel=="wwl"){
  H <- as.integer(args[3])
  T <- as.integer(args[4])
  hparams <- list("H"=H, "T"=T)
} else if (kernel=="swwl" || kernel=="aswwl"){
  H <- as.integer(args[3])
  T <- as.integer(args[4])
  P <- as.integer(args[5])
  Q <- as.integer(args[6])
  hparams <- list("H"=H, "T"=T, "P"=P, "Q"=Q)
} else if (kernel=="propag"){
  tmax <- as.integer(args[3])
  w <- as.double(args[4])
  hparams <- list("tmax"=tmax, "w"=w)
}

library(RobustGaSP)
library(reticulate)
library(yaml)

config <- yaml.load_file("./config.yml")
use_python(config$R$path_to_python)
np <- import("numpy")

load_data <- function(dataset_name, hparams, kernel, seed){
  # This function loads the data (train/test distance/Gram matrices, train/test input scalar matrices, train/test scalar outputs).

  if (dataset_name %in% c("Rotor37", "Rotor37_CM")){
    N_train <- 1000
    N_test <- 200
    out <- 3
  } else if(dataset_name %in% c("Tensile2d", "Tensile2d_CM")){
    N_train <- 500
    N_test <- 200
    out <- 1
  } else { # airfRANS or airfRANS_CM
    N_train <- 800
    N_test <- 200
    out <- 1
  }
  N <- (N_train+N_test)

  if (kernel=="wwl"){
    filenameD <- file.path(config$results$distances, dataset_name, "wwl/exp0", paste0("D_", dataset_name, "_wwl_H", hparams$H, "_T", hparams$T, "_seed0.npy"), fsep="/")
    num_powexp <- 1
  } else if (kernel=="swwl"){
    filenameD <- file.path(config$results$distances, dataset_name, "swwl", paste0("exp", seed), paste0("/D_", dataset_name, "_swwl_H", hparams$H, "_P", hparams$P, "_Q", hparams$Q, "_T", hparams$T, "_seed", seed, ".npy"), fsep="/")
    num_powexp <- 1
  } else if (kernel=="aswwl"){
    filenameD <- file.path(config$results$distances, dataset_name, "aswwl", paste0("exp", seed), paste0("/D_", dataset_name, "_aswwl_H", hparams$H, "_P", hparams$P, "_Q", hparams$Q, "_T", hparams$T, "_seed", seed, ".npy"), fsep="/")
    num_powexp <- hparams$H + 1
  }
  else if (kernel=="propag"){
    w <- "0.01"
    if (hparams$w==1e-2){
      w <- "0.01"
    } else if (hparams$w==1e-3){
      w <- "0.001"
    } else if (hparams$w==1e-4){
      w <- "0.0001"
    } else if (hparams$w==1e-5){
      w <- "1e-05"
    }
    filenameD <- file.path(config$results$gram, dataset_name, "propag", paste0("exp", seed), paste0("/K_", dataset_name, "_propag_tmax", hparams$tmax, "_w", w, "_seed", seed, ".npy"), fsep="/")
    num_powexp <- 0
  }

  Darray <- np$load(filenameD)
  Sarray <- np$load( file.path(config$results$scalar_matrices, dataset_name, paste0("S_", dataset_name,".npy"), fsep="/") )

  num_matern <- NROW(Sarray)
  num_subkernels <- (num_powexp + num_matern)

  R0_train <- list()
  if (kernel=="propag") {
    for (i in 1:num_matern){
      R0_train[[i]] <- Sarray[i,1:N_train,1:N_train]
    }
    R0_train[[num_matern+1]] <- Darray[1,1:N_train, 1:N_train]
  } else{
    R0_train <- list()
    for (i in 1:num_powexp){
      R0_train[[i]] <-  Darray[i,1:N_train, 1:N_train]
    }
    for (i in 1:num_matern){
      R0_train[[i+num_powexp]] <- Sarray[i,1:N_train,1:N_train]
    }
  }

  R0_test <- list()
  if (kernel=="propag") {
    for (i in 1:num_matern){
      R0_test[[i]] <- t(Sarray[i,1:N_train,(N_train+1):N])
    }
    R0_test[[num_matern+1]] <- t(Darray[1:N_train, (N_train+1):N])
  } else {
    for (i in 1:num_powexp){
      R0_test[[i]] <- t(Darray[i,1:N_train, (N_train+1):N])
    }
    for (i in 1:num_matern){
      R0_test[[i+num_powexp]] <- t(Sarray[i,1:N_train,(N_train+1):N])
    }
  }

  Y <- np$load( file.path(config$results$output_scalars, dataset_name, paste0("y_", dataset_name, ".npy")) )
  Y_train <- Y[1:N_train, out]
  Y_test <- Y[(N_train+1):N, out]

  X_train <- matrix(NA,N_train,num_subkernels)
  for (i in 1:num_subkernels){
    X_train[,i] <- seq(0,max(R0_train[[i]]),length.out=N_train)
  }
  # R0: stacked distance/Gram matrices and input scalar matrices.
  # X: placeholders for RGaSP to determine the bounds.  
  return(list("X_train"=X_train,"Y_train"=Y_train,"Y_test"=Y_test, "R0_train"=R0_train, "R0_test"=R0_test, "N_test"= N_test, "num_powexp"=num_powexp, "num_matern"=num_matern, "out"=out))
}

train_and_test <- function(data, seed, nugget.est=FALSE, nugget=0.00001, num_initial_values=3){
  # This function uses RGaSP to train and test using the data loaded thanks to load_data. RMSE and Q2 scores are returned.

  num_subkernels <- data$num_powexp + data$num_matern
  kernel_type <- c(rep("pow_exp", times=data$num_powexp), rep("matern_5_2", times=data$num_matern))
  kmodel<- rgasp(design=data$X_train, response=data$Y_train, kernel_type=kernel_type, 
                             R0=data$R0_train, isotropic=FALSE, nugget.est=nugget.est, nugget=nugget, num_initial_values=num_initial_values, alpha=1.99, seed=seed)
  Y_pred <- predict(kmodel, matrix(runif(data$N_test*num_subkernels),data$N_test,num_subkernels), r0=data$R0_test)$mean
  Q2 <- 1 - sum((Y_pred-data$Y_test)^2)/sum((data$Y_test-mean(data$Y_test))^2)
  RMSE <- sqrt(mean((data$Y_test - Y_pred)^2))
  return(list("RMSE"=RMSE, "Q2"=Q2))
}

max_seed <- 4 # We consider seeds 0,1,2,3,4.
nugget.est <- TRUE # Estimate the nugget?
nugget <- 0.0 # Initial nugget.
num_initial_values <- 3 # Number of restarts of RGaSP

q2s <- list()
rmses <- list()
times <- list()
for (seed in 0:max_seed){
  ptm <- proc.time()[[3]]
  i <- seed+1
  data <- load_data(dataset_name, hparams, kernel, seed)
  scores <- train_and_test(data, seed, nugget.est=nugget.est, nugget=nugget, num_initial_values=num_initial_values)
  print(scores)
  
  q2s[[i]] <- scores$Q2
  rmses[[i]] <- scores$RMSE
  times[[i]] <- (proc.time()[[3]] - ptm)
}

q2s_vec <- unlist(q2s)  # The list of Q2 scores for all seeds.
rmses_vec <- unlist(rmses) # The list of RMSE scores for all seeds.
times_vec <- unlist(times) # The list of times for all seeds.

suffix_str_scores_and_times <- function(dataset_name, kernel, hparams, out){
  # This function is a utility function to get the name of the files where to save the score and time files.
  beginning <- paste0("_", dataset_name, "_rgaspR_", kernel)
  if (kernel=="wwl"){
    suffix <- paste0(beginning, "_H", hparams$H, "_T", hparams$T, "_out", out, ".npy")
  } else if (kernel=="swwl"){
    suffix <- paste0(beginning, "_H", hparams$H, "_P", hparams$P, "_Q", hparams$Q, "_T", hparams$T, "_out", out, ".npy")
  } else if (kernel=="aswwl"){
    suffix <- paste0(beginning, "_H", hparams$H, "_P", hparams$P, "_Q", hparams$Q, "_T", hparams$T, "_out", out, ".npy")
  }
  else if (kernel=="propag"){
    w <- "0.01"
    if (hparams$w==1e-2){
      w <- "0.01"
    } else if (hparams$w==1e-3){
      w <- "0.001"
    } else if (hparams$w==1e-4){
      w <- "0.0001"
    } else if (hparams$w==1e-5){
      w <- "1e-05"
    }
    suffix <- paste0(beginning, "_tmax", hparams$tmax, "_w", w, ".npy")
  }
  return(suffix)
}

res_q2s_rmses <- array(c(q2s_vec, rmses_vec), dim=c((max_seed+1),2))
dir.create(config$results$scores, showWarnings = FALSE)
dir.create(file.path(config$results$scores, dataset_name, fsep="/"), showWarnings = FALSE)
dir.create(file.path(config$results$scores, dataset_name, kernel, fsep="/"), showWarnings = FALSE)
np$save(file.path(config$results$scores, dataset_name, kernel, paste0("scores", suffix_str_scores_and_times(dataset_name, kernel, hparams, data$out-1)), fsep="/"), res_q2s_rmses)

dir.create(config$results$times, showWarnings = FALSE)
dir.create(file.path(config$results$times, dataset_name, fsep="/"), showWarnings = FALSE)
dir.create(file.path(config$results$times, dataset_name, kernel, fsep="/"), showWarnings = FALSE)
np$save(file.path(config$results$times, dataset_name, kernel, paste0("times_training", suffix_str_scores_and_times(dataset_name, kernel, hparams, data$out-1)), fsep="/"), times_vec)

cat(dataset_name, kernel, "\n")
cat(unlist(hparams), "\n")
cat("Nugget.est =", nugget.est, ", nugget =", nugget, "\n")

cat("Times", times_vec, "\n")

cat("Final results (5 exp)", "\n")
cat("Q2s =", q2s_vec, "\n")
cat(", RMSEs =", rmses_vec, "\n")
cat(paste("Q2 =", mean(q2s_vec), "+-", sd(q2s_vec),  ", RMSE =", mean(rmses_vec), "+-", sd(rmses_vec)))