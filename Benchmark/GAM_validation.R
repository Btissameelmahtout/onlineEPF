

packages <- c("mgcv", "dplyr", "ggplot2", "reticulate")

installed <- rownames(installed.packages())

for (pkg in packages) {
  if (!pkg %in% installed) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  
  library(pkg, character.only = TRUE)
}

set.seed(42)


country <- "Germany"
data <- read.csv(paste0("../Data/",country, ".csv"))

# select the price and time
id_select <- 1
price <- data[[id_select + 1]]  # Python iloc[:,1] = R column 2

time_utc <- as.POSIXct(
  data$time_utc,
  tz = "UTC",
  format = "%Y-%m-%d %H:%M:%S"
)

local_time_zone <- "CET"
time_lt <- as.POSIXct(format(time_utc, tz = local_time_zone, usetz = TRUE))

S <- 24

# Save the start and end-time
start_end_time_S <- as.POSIXct(
  format(time_lt[c(1, length(time_lt))], tz = "UTC"),
  tz = "UTC"
)

# creating 'fake' local time
start_end_time_S_num <- as.numeric(start_end_time_S) * 1e9

time_S_numeric <- seq(
  from = start_end_time_S_num[1],
  to = start_end_time_S_num[2] + 24 * 60 * 60 * 1e9 / S,
  by = 24 * 60 * 60 * 1e9 / S
)



# 'fake' local time
time_S <- as.POSIXct(
  time_S_numeric / 1e9,
  origin = "1970-01-01",
  tz = "UTC"
)

dates_S <- unique(as.Date(time_S, tz = local_time_zone))

# Load function
source("../Functions/DST.trafo.R")

# import DST_trafo function and use it on data
data_array <- DST.trafo(
  X = data[, 2:ncol(data)],
  Xtime = time_utc,
  Xtz = local_time_zone
)





# Save the variable names
reg_names <- colnames(data)[2:ncol(data)]

test_period <- 730




# Forecast horizon
h <- 1

# Update model every observations
update_every <- 1





# Keep the last 2 years for test
N <- 2 * 365


n_days <- dim(data_array)[1]

dat_val <- data_array[1:(n_days - N), , , drop = FALSE]
days_val <- dates_S[1:(n_days - N)]



get_lagged_dt <- function(z, lags = 1, give_names = TRUE) {
  data.table::data.table(z)[
    ,
    data.table::shift(.SD, lags, give.names = give_names),
    .SDcols = seq_len(ifelse(is.null(dim(z)[2]), 1, dim(z)[2]))
  ]
}


expert_wd = c(2, 7, 1)
price_s_lags = c(1, 2, 7)
fuel_lags = c(2)

  
dimnames(dat_val)[[3]] <- reg_names
  
S <- dim(dat_val)[2]
  
  # Define object to store forecast
forecast <- numeric(S)
  
  # Get days vector incl day to forecast
days_ext <- days_val
  
  # Prepare weekday dummies
weekdays_num <- lubridate::wday(days_ext) ## 1==Sun, 2==Mon, etc.
wd <- t(sapply(weekdays_num, "==", 1:7)) + 0
dimnames(wd) <- list(
    NULL,
    c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
  )
  
  # External regressors - names for subsetting:
da_forecast_names <- c("Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA")

fuel_names <- c("Coal", "NGas", "Oil", "EUA")
  

mat_fuels <- get_lagged_dt(dat_val[, 1, fuel_names], fuel_lags)
  

  
  # Get price of last hour of yesterday
price_last <- matrix(dat_val[, S, "Price"]) # get price
price_last <- get_lagged_dt(price_last, 1) # lag it
  

  
dimnames(price_last) <- list(NULL, "Price_last") # define names
  
hyper_para <- function(
  training_wind,
  k1, k2, k3, k4,
  k5, k6, k7, k8,
  k9, k10, k11, k12
  ) {
  
  all_results <- vector("list", S)
  names(all_results) <- paste0("s", 1:S)
  
  for (s in 1:S) {
    # test: s = 1
    
    # Get actual price
    acty <- matrix(dat_val[, s, "Price"], dimnames = list(NULL, "Price"))
    
    # Get price lags
    mat_price_lags <- get_lagged_dt(acty, price_s_lags)

    
    # Get DA forecasts
    mat_da_forecasts <- dat_val[, s, da_forecast_names]
    
    # Create full reg matrix
    if (s==24) {
      # Create full reg matrix
      regmat <- as.data.frame(
        cbind(
          y = dat_val[, s, "Price"], # response is first col of reg matrix
          wd[, expert_wd],
          mat_price_lags,
          #price_last,
          mat_da_forecasts,
          mat_fuels
        )
      )
      
    } else {
      # Create full reg matrix
      regmat <- as.data.frame(
        cbind(
          y = dat_val[, s, "Price"], # response is first col of reg matrix
          wd[, expert_wd],
          mat_price_lags,
          price_last,
          mat_da_forecasts,
          mat_fuels
        )
      )
      
    }
    
    # Index without NA and without last row (since it corresponds to oos price)
    act_index <- which((rowSums(is.na(regmat)) == 0))
    
    #remove na
    regmat <- regmat[act_index, ]


# ------------------------------------------------------------
# 3. Train / validation / online-update split
# ------------------------------------------------------------


  df_s <- regmat
    
  test_start <- nrow(df_s) - test_period + 1
  initial_train_end <- test_start - 1
      
  train_initial <- df_s[
    (initial_train_end - training_wind + 1):initial_train_end,
  ]
  
  test_stream <- df_s[test_start:nrow(df_s), ]
  
  #dummy variables
  linear_vars <- c("Mon", "Sat", "Sun")

  preds <- colnames(regmat[-1])

  n_unique <- sapply(train_initial[, preds, drop = FALSE], function(x) {
    length(unique(x[!is.na(x)]))
  })


  
  
  smooth_vars <- setdiff(
    names(n_unique)[n_unique > 5],
    linear_vars
  )
  
  linear_extra <- setdiff(
    names(n_unique)[n_unique >= 2 & n_unique <= 5],
    linear_vars
  )
  
  k_by_var <- c(
    Price_lag_1 = k1,
    Price_lag_2 = k2,
    Price_lag_7 = k3,
    Price_last  = k4,
    Load_DA     = k5,
    Solar_DA    = k6,
    WindOn_DA   = k7,
    WindOff_DA  = k8,
    Coal_lag_2  = k9,
    NGas_lag_2  = k10,
    Oil_lag_2   = k11,
    EUA_lag_2   = k12
  )

  k_for_var <- k_by_var[smooth_vars]
  
  smooth_terms <- paste0(
    "s(", smooth_vars,
    ", k = ", pmin(k_for_var, n_unique[smooth_vars] - 1),
    ")"
  )
  
  if (any(is.na(k_for_var))) {
    stop("Missing k for: ", paste(smooth_vars[is.na(k_for_var)], collapse = ", "))
  }

  gam_formula <- as.formula(
    paste(
      "y ~",
      paste(c(linear_vars, linear_extra, smooth_terms), collapse = " + ")
    )
  )

  
  
  # ------------------------------------------------------------
  # 4. Fit initial GAM using bam()
  # ------------------------------------------------------------
  
  
  
  
  model <- bam(
    formula = gam_formula,
    data = train_initial,
    family = gaussian(link = "identity"),
    method = "fREML",
    discrete = FALSE
  )

  
  
  # ------------------------------------------------------------
  # 5. Rolling forecasting with bam.update()
  # ------------------------------------------------------------
  
  results <- data.frame()
  
  current_model <- model
  update_buffer <- data.frame()
  
  for (i in seq_len(nrow(test_stream))) {
    
    new_obs <- test_stream[i, , drop = FALSE]
    
    # One-step-ahead forecast
    pred <- predict(
      current_model,
      newdata = new_obs,
      type = "response",
      se.fit = TRUE
    )
    
    results <- rbind(
      results,
      data.frame(
        #date = new_obs$date,
        y_true = new_obs$y,
        y_pred = as.numeric(pred$fit),
        se = as.numeric(pred$se.fit)
      )
    )
    
    # Add observed point to update buffer
    update_buffer <- rbind(update_buffer, new_obs)
    
    # Update model in batches
    if (nrow(update_buffer) >= update_every) {
      
      current_model <- bam.update(
        current_model,
        data = update_buffer
      )
      
      update_buffer <- data.frame()
    }
  }

  all_results[[s]] <- results
  }  
  overall_rmse <- sqrt(
    mean(
      unlist(
        lapply(all_results, function(df) {
          (df$y_true - df$y_pred)^2
        })
      ),
      na.rm = TRUE
    )
  )
  return(overall_rmse)
}  





optuna <- import("optuna")

objective <- function(trial) {
  training_wind <- trial$suggest_int("training_wind", 100L, 730L)
  
  k1  <- trial$suggest_int("k1", 5L, 15L)
  k2  <- trial$suggest_int("k2", 5L, 15L)
  k3  <- trial$suggest_int("k3", 5L, 15L)
  k4  <- trial$suggest_int("k4", 5L, 15L)
  k5  <- trial$suggest_int("k5", 5L, 15L)
  k6  <- trial$suggest_int("k6", 5L, 15L)
  k7  <- trial$suggest_int("k7", 5L, 15L)
  k8  <- trial$suggest_int("k8", 5L, 15L)
  k9  <- trial$suggest_int("k9", 5L, 15L)
  k10 <- trial$suggest_int("k10", 5L, 15L)
  k11 <- trial$suggest_int("k11", 5L, 15L)
  k12 <- trial$suggest_int("k12", 5L, 15L)
  
  rmse <- tryCatch(
    {
      hyper_para(
        training_wind,
        k1, k2, k3, k4,
        k5, k6, k7, k8,
        k9, k10, k11, k12
      )
    },
    error = function(e) {
      print(e$message)
      1e9
    }
  )
  
  return(as.numeric(rmse))
}

n_trials <- 500L
sampler <- optuna$samplers$TPESampler(seed = 42L)

study <- optuna$create_study(
  direction = "minimize",
  sampler = sampler
)

study$optimize(objective, n_trials = n_trials)

best_value <- study$best_value
best_params <- study$best_params

print(best_value)
print(best_params)




saveRDS(
  list(
    best_value = best_value,
    best_params = best_params
  ),
  file = paste0(country, "_best_gam_hyperparameters.rds")
)
