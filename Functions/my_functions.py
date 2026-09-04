import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from calendar import day_abbr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.linear_model import LinearRegression
from pathlib import Path
import joblib
from statsmodels.tsa.stattools import adfuller
import os
import time
import statsmodels.api as sm
import plotly.graph_objects as go

def DST_trafo(X, Xtime, tz="CET"):
    """Converts a time series DataFrame to a DST-adjusted array

    The function takes a DataFrame of D*S rows and N columns and returns
    an array of shape (D,S,N) where D is the number of days, S the number
    of observations per day and N the number of variables. The function deals
    with the DST problem by averaging the additional hour in October and
    interpolating the missing hour in March.

    Parameters
    ----------
    X : DataFrame
        The time series DataFrame of shape (D*S,N) to be DST-adjusted.
    Xtime : datetime Series
        The series of length D*S containing UTC dates corresponding to the
        DataFrame X.
    tz : str
        The timezone to which the data needs to be adjusted to. The current
        implementation was not tested with other timezones than CET.

    Returns
    -------
    ndarray
        an ndarray of DST-adjusted variables of shape (D,S,N).
    """
    Xinit = X.values
    if len(Xinit.shape) == 1:
        Xinit = np.reshape(Xinit, (len(Xinit), 1))

    atime_init = pd.to_numeric(Xtime)
    freq = atime_init.diff().value_counts().idxmax()
    S = int(24 * 60 * 60 * 10**9 / freq)
    atime = pd.DataFrame(
        np.arange(start=atime_init.iloc[0], stop=atime_init.iloc[-1] + freq, step=freq)
    )
    idmatch = atime.reset_index().set_index(0).loc[atime_init, "index"].values
    X = np.empty((len(atime), Xinit.shape[1]))
    X[:] = np.nan
    X[idmatch] = Xinit

    new_time = Xtime.dt.tz_convert(tz).reset_index(drop=True)
    DLf = new_time.dt.strftime("%Y-%m-%d").unique()
    days = pd.Series(pd.to_datetime(DLf))

    # EUROPE
    DST_SPRING = pd.to_numeric(days.dt.strftime("%m%w")).eq(30) & pd.to_numeric(
        days.dt.strftime("%d")
    ).ge(25)
    DST_FALL = pd.to_numeric(days.dt.strftime("%m%w")).eq(100) & pd.to_numeric(
        days.dt.strftime("%d")
    ).ge(25)
    DST = ~(DST_SPRING | DST_FALL)

    time_start = new_time.iloc[range(S + int(S / 24))].dt.strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )
    time_end = new_time.iloc[range(-S - int(S / 24), 0)].dt.strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )

    Dlen = len(DLf)
    Shift = 2  # for CET

    X_dim = X.shape[1]

    Xout = np.empty((Dlen, S, X_dim))
    Xout[:] = np.nan

    k = 0
    # first entry:
    i_d = 0
    idx = time_start[time_start.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[
            i_d,
            S - 1 - idx[::-1],
        ] = X[
            range(k, len(idx) + k),
        ]
    elif DST_SPRING[i_d]:
        tmp = S - 1 - idx[::-1]
        # MARCH
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S / 24 - 1:
                Xout[
                    i_d,
                    int(S - S / 24 - len(idx) + i_S),
                ] = X[
                    k + i_S,
                ]
            if tmp[i_S] == Shift * S / 24 - 1:
                Xout[
                    i_d,
                    range(
                        int(S - S / 24 - len(idx) + i_S + 1),
                        int(S - S / 24 - len(idx) + i_S + 1 + S / 24),
                    ),
                ] = X[
                    [
                        k + i_S,
                    ]
                ] + np.transpose(
                    np.atleast_2d(
                        np.arange(1, int(S / 24) + 1) / (len(range(int(S / 24))) + 1)
                    )
                ).dot(
                    X[
                        [
                            k + i_S + 1,
                        ]
                    ]
                    - X[
                        [
                            k + i_S,
                        ]
                    ]
                )
            if tmp[i_S] > Shift * S / 24 - 1:
                Xout[
                    i_d,
                    int(S - S / 24 - len(idx) + i_S + S / 24),
                ] = X[
                    k + i_S,
                ]
    else:
        tmp = S - idx[::-1]
        # OCTOBER
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S / 24 - 1:
                Xout[
                    i_d,
                    int(S + S / 24 - len(idx) + i_S),
                ] = X[
                    k + i_S,
                ]
            if tmp[i_S] in (Shift * S / 24 - 1 + np.arange(1, int(S / 24) + 1)):
                Xout[
                    i_d,
                    int(S + S / 24 - len(idx) + i_S),
                ] = 0.5 * (
                    X[k + i_S,] + X[int(k + i_S + S / 24),]
                )
            if tmp[i_S] > (Shift + 2) * S / 24 - 1:
                Xout[
                    i_d,
                    int(S + S / 24 - len(idx) + i_S - S / 24),
                ] = X[
                    k + i_S,
                ]
    k += len(idx)
    for i_d in range(1, len(DLf) - 1):
        if DST[i_d]:
            idx = S
            Xout[
                i_d,
                range(idx),
            ] = X[
                range(k, k + idx),
            ]
        elif DST_SPRING[i_d]:
            idx = int(S - S / 24)
            # MARCH
            for i_S in range(idx):
                if i_S <= Shift * S / 24 - 1:
                    Xout[
                        i_d,
                        i_S,
                    ] = X[
                        k + i_S,
                    ]
                if i_S == Shift * S / 24 - 1:
                    Xout[
                        i_d,
                        range(int(i_S + 1), int(i_S + 1 + S / 24)),
                    ] = X[
                        [
                            k + i_S,
                        ]
                    ] + np.transpose(
                        np.atleast_2d(
                            np.arange(1, int(S / 24) + 1)
                            / (len(range(int(S / 24))) + 1)
                        )
                    ).dot(
                        X[
                            [
                                k + i_S + 1,
                            ]
                        ]
                        - X[
                            [
                                k + i_S,
                            ]
                        ]
                    )
                if i_S > Shift * S / 24 - 1:
                    Xout[
                        i_d,
                        int(i_S + S / 24),
                    ] = X[
                        k + i_S,
                    ]
        else:
            idx = int(S + S / 24)
            # October
            for i_S in range(idx):
                if i_S <= Shift * S / 24 - 1:
                    Xout[
                        i_d,
                        i_S,
                    ] = X[
                        k + i_S,
                    ]
                if i_S in (Shift * S / 24 - 1 + np.arange(1, int(S / 24) + 1)):
                    Xout[
                        i_d,
                        i_S,
                    ] = 0.5 * (
                        X[k + i_S,] + X[int(k + i_S + S / 24),]
                    )
                if i_S > (Shift + 2) * S / 24 - 1:
                    Xout[
                        i_d,
                        int(i_S - S / 24),
                    ] = X[
                        k + i_S,
                    ]
        k += idx
    # last
    i_d = len(DLf) - 1
    idx = time_end[time_end.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[
            i_d,
            range(len(idx)),
        ] = X[
            range(k, k + len(idx)),
        ]
    elif DST_SPRING[i_d]:
        # MARCH
        for i_S in range(len(idx)):
            if i_S <= Shift * S / 24 - 1:
                Xout[
                    i_d,
                    i_S,
                ] = X[
                    k + i_S,
                ]
            if i_S == Shift * S / 24 - 1:
                Xout[
                    i_d,
                    range(int(i_S + 1), int(i_S + 1 + S / 24)),
                ] = X[
                    [
                        k + i_S,
                    ]
                ] + np.transpose(
                    np.atleast_2d(
                        np.arange(1, int(S / 24) + 1) / (len(range(int(S / 24))) + 1)
                    )
                ).dot(
                    X[
                        [
                            k + i_S + 1,
                        ]
                    ]
                    - X[
                        [
                            k + i_S,
                        ]
                    ]
                )
            if i_S > Shift * S / 24 - 1:
                Xout[
                    i_d,
                    int(i_S + S / 24),
                ] = X[
                    k + i_S,
                ]
    else:
        # OCTOBER
        for i_S in range(len(idx)):
            if i_S <= Shift * S / 24 - 1:
                Xout[
                    i_d,
                    i_S,
                ] = X[
                    k + i_S,
                ]
            if i_S in (Shift * S / 24 - 1 + np.arange(1, int(S / 24) + 1)):
                Xout[
                    i_d,
                    i_S,
                ] = 0.5 * (
                    X[k + i_S,] + X[int(k + i_S + S / 24),]
                )
            if i_S > (Shift + 2) * S / 24 - 1:
                Xout[
                    i_d,
                    int(i_S - S / 24),
                ] = X[
                    k + i_S,
                ]
    return Xout


#------------------------------------------------------
#                  Regression matrix
#------------------------------------------------------


def reg_matrix(
    dat_eval, days_eval, country, wd, reg_names, fuel_lags, price_s_lags, da_lag
):
    # dat.shape[0] = D+1, days.shape[0] = D
    S = dat_eval.shape[1]

    days_ext = days_eval
    # preparation of weekday dummies
    weekdays_num = days_ext.dt.weekday + 1  # 1 = Mon, 2 = Tue, ..., 7 = Sun
    WD = np.transpose([(weekdays_num == x) + 0 for x in wd])

    # Create column names for weekdays
    wd_columns = [f"WD_{x}" for x in wd]

    # Names for subsetting:
    if country == "Germany":
        da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]

    elif country == "Spain":
        da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA"]

    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    # preparation of lags:sigmoi
    def get_lagged(Z, lag):
        return np.concatenate((np.repeat(np.nan, lag), Z[: (len(Z) - lag)]))

    # Lag of 2 as end of day data... at d-1
    mat_fuels = np.concatenate(
        [
            np.apply_along_axis(
                get_lagged, 0, dat_eval[:, 0, reg_names.isin(fuel_names)], lag=l
            )
            for l in fuel_lags
        ],
        axis=-1,
    )

    # Create column names for each fuel lag
    fuel_columns = [f"{fuel}_lag_{l}" for l in fuel_lags for fuel in fuel_names]

    price_last = get_lagged(Z=dat_eval[:, S - 1, reg_names == "Price"][..., 0], lag=1)

    price_last_column = ["price_last"]

    # Create the base matrix with shared variables outside the loop
    base_regmat = np.column_stack((WD, mat_fuels, price_last))
    column_base = wd_columns + fuel_columns + price_last_column
    regmat1 = pd.DataFrame(base_regmat, columns=column_base)
    columns_base = regmat1.shape[1]

    all_dataframes = []
    for s in range(S):

        # prepare the Y vectoracty
        acty = dat_eval[:, s, reg_names == "Price"][..., 0]

        # get lags
        mat_price_lags = np.transpose(
            [get_lagged(lag=lag, Z=acty) for lag in price_s_lags]
        )
        mat_da_forecasts = dat_eval[:, s, reg_names.isin(da_forecast_names)]

        # Create up to lag 2 for day ahead variables

        stacked_da = []
        for i in range(len(da_forecast_names)):
            da_var = np.transpose(
                [get_lagged(lag=lag, Z=mat_da_forecasts[:, i]) for lag in da_lag]
            )
            stacked_da.append(da_var)  # Store the 2D arrays

        # Stack them side by side
        da_all_var = np.hstack(stacked_da)

        # components for this s
        # regmat2 = np.column_stack((acty, mat_price_lags, mat_da_forecasts))
        regmat2 = np.column_stack((acty, mat_price_lags, da_all_var))

        # Convert regmat2 to DataFrame and rename columns to end with '_s'
        columns = (
            [f"Price_s{s}"]
            + [f"Price_lag_{lag}_s{s}" for lag in price_s_lags]
            # + [f"{name}_s{s}" for name in da_forecast_names]
            + [f"{name}_lag_{lag}_s{s}" for name in da_forecast_names for lag in da_lag]
        )
        df = pd.DataFrame(regmat2, columns=columns)

        columns_s = df.shape[1]
        # Append to list
        all_dataframes.append(df)

    # Concatenate all DataFrames
    final_dataframe = pd.concat(all_dataframes, axis=1)

    regmat = pd.concat([final_dataframe, regmat1], axis=1)
    columns_total = regmat.shape[1]
    return [regmat, columns_s, columns_base, columns_total, len(da_forecast_names)]


#------------------------------------------------------
#               function to forecast OLS
#------------------------------------------------------

def forecast_expert_ext(
    dat, days, country, reg_names, wd, price_s_lags, fuel_lags, da_lag
):

    S = dat.shape[1]
    forecast = np.repeat(np.nan, S)


    days_ext = days
    weekdays_num = days_ext.dt.weekday + 1
    WD = np.transpose([(weekdays_num == x) + 0 for x in wd])

    # Names for subsetting:
    # da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA"]
    if country == "Germany":
        da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]

    elif country == "Spain":
        da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA"]

    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    # preparation of lags:
    def get_lagged(Z, lag):
        return np.concatenate((np.repeat(np.nan, lag), Z[: (len(Z) - lag)]))

    # Lag of 2 as end of day data... at d-1
    mat_fuels = np.concatenate(
        [
            np.apply_along_axis(
                get_lagged, 0, dat[:, 0, reg_names.isin(fuel_names)], lag=l
            )
            for l in fuel_lags
        ],
        axis=-1,
    )
    price_last = get_lagged(Z=dat[:, S - 1, reg_names == "Price"][..., 0], lag=1)

    coefs = np.empty(
        (
            S,
            len(wd)
            + len(price_s_lags)
            + len(fuel_names) * len(fuel_lags)
            + len(da_forecast_names) * len(da_lag)
            + 1,
        )
    )

    for s in range(S):
        # prepare the Y vector
        acty = dat[:, s, reg_names == "Price"][..., 0]

        # get lags
        mat_price_lags = np.transpose(
            [get_lagged(lag=lag, Z=acty) for lag in price_s_lags]
        )
        mat_da_forecasts = dat[:, s, reg_names.isin(da_forecast_names)]

        # Create up to lag 2 for day ahead variables

        stacked_da = []
        for i in range(len(da_forecast_names)):
            da_var = np.transpose(
                [get_lagged(lag=lag, Z=mat_da_forecasts[:, i]) for lag in da_lag]
            )
            stacked_da.append(da_var)  # Store the 2D arrays

        # Stack them side by side
        da_all_var = np.hstack(stacked_da)
        # remove price_last for last hour as it is equal to price of yesterday
        if s == S - 1:
            regmat = np.column_stack(
                (
                    acty,
                    # np.ones(acty.shape[0]),
                    mat_price_lags,
                    da_all_var,
                    WD,
                    mat_fuels,
                )
            )
        else:
            # combine all regressors to a matrix
            regmat = np.column_stack(
                (
                    acty,
                    # np.ones(acty.shape[0]),
                    mat_price_lags,
                    da_all_var,
                    WD,
                    mat_fuels,
                    price_last,
                )
            )

        # drop all rows with NAs
        act_index = ~np.isnan(regmat).any(axis=1)
        regmat0 = regmat[act_index]

        # scaling using the training mean and std of trainig data only
        regmat_mean = regmat0[:-1, :].mean(axis=0)
        regmat_sd = regmat0[:-1, :].std(axis=0)
        regmat_sd[regmat_sd == 0] = 1
        regmat_scaled = (regmat0 - regmat_mean) / regmat_sd

        model = LinearRegression(fit_intercept=False).fit(
            X=regmat_scaled[:-1, 1:], y=regmat_scaled[:-1, 0]
        )

        # deal with singularities
        model.coef_[np.isnan(model.coef_)] = 0

        forecast[s] = (
            (model.coef_ @ regmat_scaled[-1, 1:]) * regmat_sd[0]
        ) + regmat_mean[0]

        if s == S - 1:
            coefs[s] = np.append(model.coef_, 0)
        else:
            coefs[s] = model.coef_

    regressor_names = (
        # ["intercept"]+
        ["Price lag " + str(lag) for lag in price_s_lags]
        + [f"{name}_lag_{lag}_s{s}" for name in da_forecast_names for lag in da_lag]
        + [day_abbr[i - 1] for i in wd]
        + [fuel + " lag " + str(lag) for lag in fuel_lags for fuel in fuel_names]
        + ["Price last lag 1"]
    )

    coefs_df = pd.DataFrame(coefs, columns=regressor_names)

    return {"forecasts": forecast, "coefficients": coefs_df}


# %%--------------------------------------------------
#                 L1  regulization term
#-------------------------------------------------------


def l1_regularization(model, model_type, lambda_reg):
    l1_norm = 0.0

    if model_type in {1, 4, 7}:
        for param in model.linear.parameters():
            if param.requires_grad:
                l1_norm += torch.sum(torch.abs(param))

    elif model_type == 2:
        for param in model.mid_to_out.parameters():
            if param.requires_grad:
                l1_norm += torch.sum(torch.abs(param))

    else:
        for param in model.mid_to_out.parameters():
            if param.requires_grad:
                l1_norm += torch.sum(torch.abs(param))

        for param in model.input_to_output.parameters():
            if param.requires_grad:
                l1_norm += torch.sum(torch.abs(param))

    reg = lambda_reg * l1_norm
    return reg


#------------------------------------------------------
#                   BOA Function
#------------------------------------------------------


def boa_fully_adaptive(M, expert_preds, actuals, length_study, device, w0=None):
    """
    Fully Adaptive Bernstein Online Aggregation (BOA)

    This function implements the fully adaptive BOA algorithm for sequential
    forecast combination. The method dynamically updates expert weights using
    second-order regret corrections, variance adaptation, and range-based
    stabilization.

    At each time step t, the algorithm:

    1. Forms an aggregate prediction as a weighted average of expert forecasts.
    2. Computes the linearized excess loss (instantaneous regret) using the
       subgradient of the chosen loss function.
    3. Updates:
        - V_t : cumulative squared regret (variance proxy)
        - E_t : running magnitude bound (range estimator)
        - R_t : corrected cumulative regret
    4. Computes adaptive learning rates balancing efficiency and stability.
    5. Updates expert weights via a SoftMax-style exponential rule.
    """
    # Move tensors to correct device (CPU/GPU)
    expert_preds = expert_preds.to(device)
    actuals = actuals.to(device)

    # ------------------------------------------------------------
    # Prior weights (uniform if not provided)
    # BOA works with probabilities → must sum to 1
    # ------------------------------------------------------------
    if w0 is None:
        w0 = torch.ones(M + 1, device=device) / (M + 1)
    else:
        w0 = w0.to(device)

    # Store weights over time
    w_history = torch.zeros((length_study + 1, M + 1), device=device)
    w_history[0] = w0

    # Aggregator predictions
    agg_preds = torch.zeros(length_study, device=device)

    # Aggregator loss tracking (diagnostics only)
    agg_loss_history = torch.zeros(length_study, device=device)

    # ------------------------------------------------------------
    # Fully adaptive BOA state variables
    # R → cumulative regret (corrected)
    # V → cumulative squared regret (variance)
    # E → running magnitude bound (range estimator)
    # ------------------------------------------------------------
    R = torch.zeros(M + 1, device=device)
    V = torch.zeros(M + 1, device=device)
    E = torch.zeros(M + 1, device=device)

    # Numerical stability offset
    tiny = 1e-9

    # Prior complexity term (appears in learning rate)
    log_w0 = torch.log(w0 + tiny)

    # ============================================================
    # Main online loop
    # ============================================================
    for t in range(length_study):

        # --------------------------------------------------------
        # (a) Aggregator prediction using current weights
        # --------------------------------------------------------
        agg_preds[t] = (w_history[t] * expert_preds[t, :]).sum()

        # --------------------------------------------------------
        # Gradient of absolute loss:
        # ∂|ŷ - y| / ∂ŷ = sign(ŷ - y)
        # --------------------------------------------------------
        grad = torch.sign(agg_preds[t] - actuals[t])

        # Gradient evaluated at aggregator prediction
        agg_regret = grad * agg_preds[t]

        # Gradient evaluated at expert predictions
        exp_regret = grad * expert_preds[t, :]

        # --------------------------------------------------------
        # Instantaneous BOA regret (linearized excess loss)
        # --------------------------------------------------------
        r_t = exp_regret - agg_regret

        # --------------------------------------------------------
        # Range estimator (protects against large updates)
        # E_t = max(E_{t-1}, |r_t|)
        # --------------------------------------------------------
        r_abs = torch.abs(r_t)
        E = torch.max(E, r_abs)

        # --------------------------------------------------------
        # Variance accumulator (second-order adaptation)
        # --------------------------------------------------------
        V = V + r_t**2

        # --------------------------------------------------------
        # Fully adaptive BOA learning rate
        # η_t ∝ sqrt( complexity / variance )
        # plus stability cap via range bound
        # --------------------------------------------------------
        eta_t = torch.sqrt(-log_w0 / (V + tiny))
        eta_t = torch.min(eta_t, 1.0 / (2.0 * (E + tiny)))

        # --------------------------------------------------------
        # Corrected regret update (Bernstein-style)
        # This is BOA's key stability refinement
        # --------------------------------------------------------
        half_step = 0.5 * r_t * (1.0 + eta_t * r_t)

        # Large-update safeguard (rare but critical)
        mask = (2.0 * eta_t * r_t) > 1.0
        indicator_term = E * mask.float()

        R = R + half_step + indicator_term

        # --------------------------------------------------------
        # Softmax-style weight update
        # BOA includes log(η_t) correction term
        # --------------------------------------------------------
        exponent = -eta_t * R + torch.log(eta_t + tiny)

        unnormalized = torch.exp(exponent)

        # Normalize → probabilities
        w_new = unnormalized / (unnormalized.sum())

        w_history[t + 1] = w_new

        # --------------------------------------------------------
        # Diagnostics: aggregator absolute loss
        # --------------------------------------------------------
        agg_loss_history[t] = torch.abs(agg_preds[t] - actuals[t])

    return w_history, agg_preds, agg_loss_history

#-------------------------------------------------
#              Neural Network for one step forecasting
#-------------------------------------------------


def train_and_evaluate_updated_weights(
    train_loader,
    test_loader,
    num_feature,
    previous_weights_s,
    num_epochs,
    learning_rate,
    number_neurons,
    std_y,
    mean_y,
    mask_input_to_mid,
    use_ols_weights,
    ols_tensor,
    weight_decay,
    alpha,
    lambda_reg,
    output_dim,
    mask_in_out_red,
    mask_in_out_full,
    device,
    model_type,
):

    class MaskedLinear(nn.Module):
        def __init__(self, input_dim, output_dim, mask):
            super(MaskedLinear, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.register_buffer("mask", mask)

        def forward(self, x):
            masked_weight = self.linear.weight * self.mask
            return nn.functional.linear(x, masked_weight, self.linear.bias)

    class MaskedLinearWithActivation(nn.Module):
        def __init__(self, input_dim, output_dim, mask, activation):
            super(MaskedLinearWithActivation, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.register_buffer("mask", mask)
            self.activation = activation

        def forward(self, x):
            masked_weight = self.linear.weight * self.mask
            outputs = nn.functional.linear(x, masked_weight, self.linear.bias)
            return self.activation(outputs)

    class CustomModelWithSkip(nn.Module):
        def __init__(
            self,
            input_dim,
            middle_dim,
            output_dim,
            mask_input_to_mid,
            mask_input_to_output,
        ):
            super(CustomModelWithSkip, self).__init__()
            # 1) Your existing masked linear with activation
            self.input_to_mid = MaskedLinearWithActivation(
                input_dim, middle_dim, mask_input_to_mid, activation=nn.LeakyReLU()
            )
            # 2) The linear layer from mid to output dimension
            self.mid_to_out = nn.Linear(middle_dim, output_dim)
            # 3) ADD a normalization layer with 'output_dim'

            # self.bn =nn.LayerNorm(output_dim)
            # 4) The skip connection from input to output
            self.input_to_output = MaskedLinear(
                input_dim, output_dim, mask_input_to_output
            )

        def forward(self, x):
            # a) Pass through masked linear + LeakyReLU
            mid_output = self.input_to_mid(x)
            # b) Map from middle_dim -> output_dim
            mid_to_output = self.mid_to_out(mid_output)
            # c) Normalize the output
            # mid_to_output = self.bn(mid_to_output)
            # d) Skip connection from input directly to output
            input_to_output = self.input_to_output(x)
            return mid_to_output + input_to_output

    class CustomModelWithoutSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid):
            super(CustomModelWithoutSkip, self).__init__()
            self.input_to_mid = MaskedLinearWithActivation(
                input_dim, middle_dim, mask_input_to_mid, activation=nn.LeakyReLU()
            )
            self.mid_to_out = nn.Linear(middle_dim, output_dim)

            # self.bn = nn.LayerNorm(output_dim)

        def forward(self, x):
            mid_output = self.input_to_mid(x)
            mid_to_output = self.mid_to_out(mid_output)
            # mid_to_output = self.bn(mid_to_output)
            return mid_to_output

    # Create the model
    input_dim = num_feature
    
    middle_dim = number_neurons

    # Intilize the model 1 and 4 with different mask tensor
    if model_type in {1, 4, 7}:
        if model_type in {1, 7}:
            mask_tensor = mask_in_out_red
        elif model_type == 4:
            mask_tensor = mask_in_out_full
        model = MaskedLinear(input_dim, output_dim, mask_tensor).to(device)

    elif model_type == 2:
        model = CustomModelWithoutSkip(
            input_dim, middle_dim, output_dim, mask_input_to_mid
        ).to(device)

    elif model_type in {3, 5, 8}:

        if model_type in {3, 8}:
            mask_tensor = mask_in_out_red
        elif model_type == 5:
            mask_tensor = mask_in_out_full

        model = CustomModelWithSkip(
            input_dim, middle_dim, output_dim, mask_input_to_mid, mask_tensor
        ).to(device)

    else:
        raise ValueError("Invalid model type. Choose 1, 2, or 3.")

    # Initialize weights
    if previous_weights_s is None:
        if model_type == 8:
            if use_ols_weights:
                model.input_to_output.linear.weight.data = alpha * ols_tensor.clone()
                model.input_to_output.linear.bias.data.zero_()  # Bias is typically set to zero

                for param in model.input_to_mid.parameters():
                    if param.requires_grad:
                        nn.init.uniform_(param, -0.001, 0.001)

                for param in model.mid_to_out.parameters():
                    if param.requires_grad:
                        nn.init.uniform_(param, -0.001, 0.001)

            else:
                for param in model.input_to_output.parameters():
                    if param.requires_grad:
                        nn.init.uniform_(param, -0.001, 0.001)  # Random initialization

        elif model_type == 7:
            if use_ols_weights:
                model.linear.weight.data = alpha * ols_tensor.clone()
                model.linear.bias.data.zero_()
            else:
                for param in model.parameters():
                    if param.requires_grad:
                        nn.init.uniform_(param, -0.001, 0.001)

        else:
            for param in model.parameters():
                if param.requires_grad:
                    nn.init.uniform_(param, -0.001, 0.001)

    else:
        # Load weights for all relevant layers in the specific model
        if model_type in {1, 4, 7}:
            model.linear.weight.data = previous_weights_s["input_to_output"][
                "weight"
            ].clone()
            model.linear.bias.data = previous_weights_s["input_to_output"][
                "bias"
            ].clone()

        elif model_type == 2:
            model.input_to_mid.linear.weight.data = previous_weights_s["input_to_mid"][
                "weight"
            ].clone()
            model.input_to_mid.linear.bias.data = previous_weights_s["input_to_mid"][
                "bias"
            ].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"][
                "weight"
            ].clone()
            model.mid_to_out.bias.data = previous_weights_s["mid_to_out"][
                "bias"
            ].clone()

        elif model_type in {3, 5, 8}:
            model.input_to_mid.linear.weight.data = previous_weights_s["input_to_mid"][
                "weight"
            ].clone()
            model.input_to_mid.linear.bias.data = previous_weights_s["input_to_mid"][
                "bias"
            ].clone()
            model.mid_to_out.weight.data = previous_weights_s["mid_to_out"][
                "weight"
            ].clone()
            model.mid_to_out.bias.data = previous_weights_s["mid_to_out"][
                "bias"
            ].clone()
            model.input_to_output.linear.weight.data = previous_weights_s[
                "input_to_output"
            ]["weight"].clone()
            model.input_to_output.linear.bias.data = previous_weights_s[
                "input_to_output"
            ]["bias"].clone()

    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Training loop
    for epoch in range(num_epochs):
        for X_train, y_train in train_loader:

            # Run a forward pass
            pred = model(X_train).squeeze(-1)
            # Compute loss and gradients
            loss = criterion(pred, y_train) + l1_regularization(
                model, model_type, lambda_reg
            )

            # Set the gradients to zero
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters
            optimizer.step()

    # Evaluation loop
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations during testing
        for X_test, y_test in test_loader:

            outputs = model(X_test).squeeze(-1)

            # Unstandardize outputs and y_test
            unstandardized_outputs = (outputs * std_y) + mean_y
            unstandardized_y_test = (y_test * std_y) + mean_y

            # calculate squared and absolute errors
            squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2
            abs_errors = torch.abs(unstandardized_outputs - unstandardized_y_test)

    # Store the final weights to use in the next iteration

    if model_type in {1, 4, 7}:

        previous_weights_s = {
            "input_to_output": {
                "weight": (model.linear.weight.data * mask_tensor).clone(),
                "bias": model.linear.bias.data.clone(),
            }
        }

    elif model_type == 2:
        previous_weights_s = {
            "input_to_mid": {
                "weight": (
                    model.input_to_mid.linear.weight.data * mask_input_to_mid
                ).clone(),
                "bias": model.input_to_mid.linear.bias.data.clone(),
            },
            "mid_to_out": {
                "weight": (model.mid_to_out.weight.data).clone(),
                "bias": model.mid_to_out.bias.data.clone(),
            },
        }

    elif model_type in {3, 5, 8}:

        previous_weights_s = {
            "input_to_mid": {
                "weight": (
                    model.input_to_mid.linear.weight.data * mask_input_to_mid
                ).clone(),
                "bias": model.input_to_mid.linear.bias.data.clone(),
            },
            "mid_to_out": {
                "weight": (model.mid_to_out.weight.data).clone(),
                "bias": model.mid_to_out.bias.data.clone(),
            },
            "input_to_output": {
                "weight": (
                    model.input_to_output.linear.weight.data * mask_tensor
                ).clone(),
                "bias": model.input_to_output.linear.bias.data.clone(),
            },
        }

    # return squared_errors, previous_weights_s, unstandardized_outputs
    return (
        squared_errors,
        abs_errors,
        previous_weights_s,
        unstandardized_y_test,
        unstandardized_outputs,
    )
#-------------------------------------------------
#     Neural Network for one step forecasting without online
#-------------------------------------------------

def train_and_evaluate_updated_weights_no_online(
    train_loader,
    test_loader,
    num_feature,
    num_epochs,
    learning_rate,
    number_neurons,
    std_y,
    mean_y,
    mask_input_to_mid,
    weight_decay,
    lambda_reg,
    mask_in_out_red,
    mask_in_out_full,
    output_dim,
    device,
    model_type,
):

    class MaskedLinear(nn.Module):
        def __init__(self, input_dim, output_dim, mask):
            super(MaskedLinear, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.register_buffer("mask", mask)

        def forward(self, x):
            masked_weight = self.linear.weight * self.mask
            return nn.functional.linear(x, masked_weight, self.linear.bias)

    class MaskedLinearWithActivation(nn.Module):
        def __init__(self, input_dim, output_dim, mask, activation):
            super(MaskedLinearWithActivation, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.register_buffer("mask", mask)
            self.activation = activation

        def forward(self, x):
            masked_weight = self.linear.weight * self.mask
            outputs = nn.functional.linear(x, masked_weight, self.linear.bias)
            return self.activation(outputs)

    class CustomModelWithSkip(nn.Module):
        def __init__(
            self,
            input_dim,
            middle_dim,
            output_dim,
            mask_input_to_mid,
            mask_input_to_output,
        ):
            super(CustomModelWithSkip, self).__init__()
            # 1) Your existing masked linear with activation
            self.input_to_mid = MaskedLinearWithActivation(
                input_dim, middle_dim, mask_input_to_mid, activation=nn.LeakyReLU()
            )
            # 2) The linear layer from mid to output dimension
            self.mid_to_out = nn.Linear(middle_dim, output_dim)
            # 3) ADD a normalization layer with 'output_dim'

            # self.bn =nn.LayerNorm(output_dim)
            # 4) The skip connection from input to output
            self.input_to_output = MaskedLinear(
                input_dim, output_dim, mask_input_to_output
            )

        def forward(self, x):
            # a) Pass through masked linear + LeakyReLU
            mid_output = self.input_to_mid(x)
            # b) Map from middle_dim -> output_dim
            mid_to_output = self.mid_to_out(mid_output)
            # c) Skip connection from input directly to output
            input_to_output = self.input_to_output(x)
            return mid_to_output + input_to_output

    class CustomModelWithoutSkip(nn.Module):
        def __init__(self, input_dim, middle_dim, output_dim, mask_input_to_mid):
            super(CustomModelWithoutSkip, self).__init__()
            self.input_to_mid = MaskedLinearWithActivation(
                input_dim, middle_dim, mask_input_to_mid, activation=nn.LeakyReLU()
            )
            self.mid_to_out = nn.Linear(middle_dim, output_dim)

            # self.bn = nn.LayerNorm(output_dim)

        def forward(self, x):
            mid_output = self.input_to_mid(x)
            mid_to_output = self.mid_to_out(mid_output)
            # mid_to_output = self.bn(mid_to_output)
            return mid_to_output

    # Create the model
    input_dim = num_feature
    
    middle_dim = number_neurons

    # Intilize the model 1 and 4 with different mask tensor
    if model_type in {1, 4, 7}:
        if model_type in {1, 7}:
            mask_tensor = mask_in_out_red
        elif model_type == 4:
            mask_tensor = mask_in_out_full
        model = MaskedLinear(input_dim, output_dim, mask_tensor).to(device)

    elif model_type == 2:
        model = CustomModelWithoutSkip(
            input_dim, middle_dim, output_dim, mask_input_to_mid
        ).to(device)

    elif model_type in {3, 5, 6, 8}:

        if model_type in {3, 6, 8}:
            mask_tensor = mask_in_out_red
        elif model_type == 5:
            mask_tensor = mask_in_out_full

        model = CustomModelWithSkip(
            input_dim, middle_dim, output_dim, mask_input_to_mid, mask_tensor
        ).to(device)

    else:
        raise ValueError("Invalid model type. Choose 1, 2, or 3.")


    for param in model.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, -0.001, 0.001)  # Random initialization

   
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


    for epoch in range(num_epochs):
        for X_train, y_train in train_loader:

            # Run a forward pass
            pred = model(X_train).squeeze(-1)
            # Compute loss and gradients
            loss = criterion(pred, y_train) + l1_regularization(
                model, model_type, lambda_reg
            )

            # Set the gradients to zero
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters
            optimizer.step()

    # Evaluation loop
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations during testing
        for X_test, y_test in test_loader:

            outputs = model(X_test).squeeze(-1)

            # Unstandardize outputs and y_test
            unstandardized_outputs = (outputs * std_y) + mean_y
            unstandardized_y_test = (y_test * std_y) + mean_y

            # calculate the squared and abs error
            squared_errors = (unstandardized_outputs - unstandardized_y_test) ** 2
            abs_errors = torch.abs(unstandardized_outputs - unstandardized_y_test)

    return (
        squared_errors,
        abs_errors,
        unstandardized_y_test,
        unstandardized_outputs,
    )



#----------------------------------------------------
#            Complet Forecasting study
#----------------------------------------------------


def forecast_MLP_rolling(
    dat_eval,
    begin_eval,
    regmat_tensor_eval,
    dependent_var_tensor_eval,
    regmat0_eval,
    days_eval,
    learning_rate_init,
    num_epochs_init,
    D_init,
    learning_rate_all,
    num_epochs_all,
    D_all,
    number_neurons,
    use_ols_weights,
    weight_decay_init,
    weight_decay_all,
    alpha,
    lambda_reg_init,
    lambda_reg_all,
    length_study,
    dependent_index,
    active_regressor,
    mask_in_out_red,
    mask_in_out_full,
    batch_size,
    wd,
    price_s_lags,
    fuel_lags,
    da_lag,
    num_columns,
    reg_names,
    country,
    device,
    model_type,
):

    previous_weights_s = None




    output_dim = dat_eval.shape[1]

    mae_all = torch.zeros((length_study, output_dim))  # GPU tensor


    

    

    mask_in_mid = torch.ones(
        (number_neurons, num_columns), dtype=torch.float32, device=device
    )
    mask_in_mid[:, dependent_index] = 0
    mask_input_to_mid = mask_in_mid


    unstandardized_outputs_all = torch.zeros((length_study, output_dim), device=device)

    unstandardized_y_test_all = torch.zeros((length_study, output_dim), device=device) 

    for n in range(length_study):

        if n == 0:
            # if n in range(0, number_init_wind+1):
            learning_rate = learning_rate_init
            num_epochs = num_epochs_init
            D = D_init
            weight_decay = weight_decay_init
            lambda_reg = lambda_reg_init

            # extract the OLS coeffiecients
            if model_type in {7, 8}:
                coef_OLS_all = forecast_expert_ext(
                    dat=dat_eval[
                        (
                            begin_eval
                            + (dat_eval.shape[0] - regmat0_eval.shape[0])
                            - D
                            + n
                        ) : (
                            begin_eval
                            + (dat_eval.shape[0] - regmat0_eval.shape[0])
                            + n
                            + 1
                        ),
                        :,
                    ],
                    days=days_eval[
                        (
                            begin_eval
                            + (dat_eval.shape[0] - regmat0_eval.shape[0])
                            - D
                            + n
                        ) : (
                            begin_eval
                            + (dat_eval.shape[0] - regmat0_eval.shape[0])
                            + n
                            + 1
                        )
                    ],
                    country=country,
                    reg_names=reg_names,
                    wd=wd,
                    price_s_lags=price_s_lags,
                    fuel_lags=fuel_lags,
                    da_lag=da_lag,
                )["coefficients"]
                coef_OLS_ten = torch.tensor(
                    coef_OLS_all.values, dtype=torch.float32, device=device
                )
                # Initialize an empty tensor to store OLS coeffiencts
                ols_tensor = torch.zeros(
                    (output_dim, num_columns), dtype=torch.float32, device=device
                )
                # Populate ols_tensor with ols coefficient
                for row_idx, col_indices in active_regressor.items():
                    ols_tensor[row_idx, col_indices] = coef_OLS_ten[
                        row_idx, : len(col_indices)
                    ]
            else:
                ols_tensor = None

        else:
            learning_rate = learning_rate_all
            num_epochs = num_epochs_all
            D = D_all
            weight_decay = weight_decay_all
            lambda_reg = lambda_reg_all

        ##%% # Scaling and standardization regmat_tensor
        mean_x = regmat_tensor_eval[(begin_eval - D + n) : (begin_eval + n),].mean(
            dim=0, keepdim=True
        )

        std_x = regmat_tensor_eval[(begin_eval - D + n) : (begin_eval + n),].std(
            dim=0, keepdim=True, unbiased=False
        )
        std_y = dependent_var_tensor_eval[(begin_eval - D + n) : (begin_eval + n),].std(
            dim=0, keepdim=True, unbiased=False
        )

        std_x[std_x == 0] = 1

        X_train = (
            regmat_tensor_eval[(begin_eval - D + n) : (begin_eval + n),] - mean_x
        ) / std_x

        # Now standardize TEST data with the SAME mean and std
        X_test = ((regmat_tensor_eval[(begin_eval + n),] - mean_x) / std_x).reshape(
            1, -1
        )

        # Scaling and standardization dependent_var_tensor_eval
        mean_y = dependent_var_tensor_eval[
            (begin_eval - D + n) : (begin_eval + n),
        ].mean(dim=0, keepdim=True)

        std_y[std_y == 0] = 1

        y_train = (
            dependent_var_tensor_eval[(begin_eval - D + n) : (begin_eval + n),] - mean_y
        ) / std_y

        # Now standardize TEST data with the SAME mean and std
        y_test = (
            (dependent_var_tensor_eval[(begin_eval + n),] - mean_y) / std_y
        ).reshape(1, -1)

        # use dataloader
        dataset_train = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        dataset_test = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        num_feature = regmat_tensor_eval.shape[1]

        ##%%

        # Train and evaluate the model
        _, mae_value, previous_weights_s, unstandardized_y_test, unstandardized_outputs = (
            train_and_evaluate_updated_weights(
                train_loader,
                test_loader,
                num_feature,
                previous_weights_s,
                num_epochs,
                learning_rate,
                number_neurons,
                std_y,
                mean_y,
                mask_input_to_mid,
                use_ols_weights,
                ols_tensor,
                weight_decay,
                alpha,
                lambda_reg,
                output_dim,
                mask_in_out_red,
                mask_in_out_full,
                device,
                model_type,
            )
        )

        mae_all[n, :] = mae_value

        # store the forecast for each n
        unstandardized_outputs_all[n,] = unstandardized_outputs.squeeze(0)

        unstandardized_y_test_all[n,] = unstandardized_y_test.squeeze(0)
    

    # Overall mean
    overall_agg_mean = mae_all.mean((0, 1))

    # Return timing details along with results
    return [
        overall_agg_mean,
        unstandardized_y_test_all,
        unstandardized_outputs_all,
    ]



#----------------------------------------------------
#        Complet Forecasting study without online learning
#----------------------------------------------------



def forecast_MLP_rolling_no_online(
    dat_eval,
    begin_eval,
    regmat_tensor_eval,
    dependent_var_tensor_eval,
    learning_rate,
    num_epochs,
    D,
    number_neurons,
    weight_decay,
    lambda_reg,
    length_study,
    dependent_index,
    mask_in_out_red,
    mask_in_out_full,
    batch_size,
    num_columns,
    device,
    model_type,
):


    output_dim = dat_eval.shape[1]

    mae_all = torch.zeros((length_study, output_dim))  # GPU tensor


    

    

    mask_in_mid = torch.ones(
        (number_neurons, num_columns), dtype=torch.float32, device=device
    )
    mask_in_mid[:, dependent_index] = 0
    mask_input_to_mid = mask_in_mid


    unstandardized_outputs_all = torch.zeros((length_study, output_dim), device=device)

    unstandardized_y_test_all = torch.zeros((length_study, output_dim), device=device) 

    for n in range(length_study):



        ##%% # Scaling and standardization regmat_tensor
        mean_x = regmat_tensor_eval[(begin_eval - D + n) : (begin_eval + n),].mean(
            dim=0, keepdim=True
        )

        std_x = regmat_tensor_eval[(begin_eval - D + n) : (begin_eval + n),].std(
            dim=0, keepdim=True, unbiased=False
        )
        std_y = dependent_var_tensor_eval[(begin_eval - D + n) : (begin_eval + n),].std(
            dim=0, keepdim=True, unbiased=False
        )

        std_x[std_x == 0] = 1

        X_train = (
            regmat_tensor_eval[(begin_eval - D + n) : (begin_eval + n),] - mean_x
        ) / std_x

        # Now standardize TEST data with the SAME mean and std
        X_test = ((regmat_tensor_eval[(begin_eval + n),] - mean_x) / std_x).reshape(
            1, -1
        )

        # Scaling and standardization dependent_var_tensor_eval
        mean_y = dependent_var_tensor_eval[
            (begin_eval - D + n) : (begin_eval + n),
        ].mean(dim=0, keepdim=True)

        std_y[std_y == 0] = 1

        y_train = (
            dependent_var_tensor_eval[(begin_eval - D + n) : (begin_eval + n),] - mean_y
        ) / std_y

        # Now standardize TEST data with the SAME mean and std
        y_test = (
            (dependent_var_tensor_eval[(begin_eval + n),] - mean_y) / std_y
        ).reshape(1, -1)

        # use dataloader
        dataset_train = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        dataset_test = TensorDataset(X_test, y_test)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        num_feature = regmat_tensor_eval.shape[1]

        ##%%

        # Train and evaluate the model
        _, mae_value, unstandardized_y_test, unstandardized_outputs = (
            train_and_evaluate_updated_weights_no_online(
                train_loader,
                test_loader,
                num_feature,
                num_epochs,
                learning_rate,
                number_neurons,
                std_y,
                mean_y,
                mask_input_to_mid,
                weight_decay,
                lambda_reg,
                mask_in_out_red,
                mask_in_out_full,
                output_dim,
                device,
                model_type,)
            )

        mae_all[n, :] = mae_value

        # store the forecast for each n
        unstandardized_outputs_all[n,] = unstandardized_outputs.squeeze(0)

        unstandardized_y_test_all[n,] = unstandardized_y_test.squeeze(0)
    

    # Overall mean
    overall_agg_mean = mae_all.mean((0, 1))

    # Return timing details along with results
    return [
        overall_agg_mean,
        unstandardized_y_test_all,
        unstandardized_outputs_all,
    ]




#-------------------------------------------------------
#            Stepwise Selection of models for BOA
#--------------------------------------------------------

def BOA_specific_combination(
    trial_list,
    dat_eval,
    length_study,
    Forecast_trials,
    actual_eval,
    device,
):

    output_dim= dat_eval.shape[1]

    # tensor to store the aggregate MSE
    agg_sq_all = torch.zeros((length_study, output_dim))



    # create a tensor to stor the forecast of the models up to best model
    forecast_tensor_eval = torch.zeros((length_study, output_dim, len(trial_list)), device=device)
    for idx, trial_num in enumerate(trial_list):
        forecast_tensor_eval[:, :, idx] = Forecast_trials[:, :, trial_num]

    M = len(trial_list) - 1
    weights_all_series = {}

    for s in range(output_dim):
        # 1) Extract the slice for series s:
        expert_preds_s = forecast_tensor_eval[:, s, :]
        #    shape (T,) for the actuals
        actuals_s = actual_eval[:, s]

        if M == 0:
            # Only one model: use weight 1.0
            wt = torch.ones((length_study, 1), device=device)
            # agg_sq = (expert_preds_s.squeeze() - actuals_s)**2
            agg_sq = torch.abs(expert_preds_s.squeeze() - actuals_s)
        else:
            wt, _, agg_sq = boa_fully_adaptive(
                M, expert_preds_s, actuals_s, length_study, device, w0=None
            )


        weights_all_series[s] = wt

        agg_sq_all[:, s] = agg_sq


    overall_agg_mean = agg_sq_all.mean((0, 1))

    # Return timing details along with results
    return overall_agg_mean, weights_all_series



def stepwise_selection(
    dat_eval, 
    actual_eval,   
    trials_df,
    best_number,
    length_study,
    Forecast_trials,
    device,
):


    # # Track model numbers excluding best  in the best 100 models
    # all_trials = list(lowest_100["number"])
    all_trials = list(trials_df["number"])
    if best_number in all_trials:
        all_trials.remove(best_number)
    weight_boa_step = {}

    # Step 0: Evaluate best_number alone
    rmse_single, weights_single = BOA_specific_combination(
        [best_number],  
        dat_eval,
        length_study,
        Forecast_trials,
        actual_eval,
        device, )
    rmse_progression = [rmse_single.item()]
    model_progression = [[best_number]]
    weight_boa_step[1] = weights_single  # 1 model → weights dict for each series

    # Step 1: Try all pairs [best_number, i]
    results_step1 = {}
    weights_step1 = {}
    for i in all_trials:
        rmse, weights = BOA_specific_combination(
        [best_number, i], 
        dat_eval,  
        length_study,
        Forecast_trials,
        actual_eval,
        device, )
        results_step1[i] = rmse
        weights_step1[i] = weights

    # Step 2: Pick best between just best_number and all pairs
    rmse_options = {tuple([best_number]): rmse_single}
    rmse_options.update(
        {(best_number, i): rmse for i, rmse in results_step1.items()}
    )

    best_models_combo, best_val = min(rmse_options.items(), key=lambda x: x[1])
    selected_models = list(best_models_combo)
    rmse_progression = [best_val.item()]
    model_progression = [selected_models.copy()]
    weight_boa_step[len(selected_models)] = (
        weights_single
        if len(selected_models) == 1
        else weights_step1[selected_models[1]]
    )

    # Step 3: Add up to max_models
    max_models = 10
    while len(selected_models) < max_models:
        print(f"\nEvaluating candidates to add to: {selected_models}")
        candidates = [i for i in all_trials if i not in selected_models]

        step_results = {}
        step_weights = {}
        for i in candidates:
            trial_list = selected_models + [i]
            rmse, weights = BOA_specific_combination(
                trial_list,
                dat_eval,   
                length_study,
                Forecast_trials,
                actual_eval,
                device, )
            step_results[i] = rmse
            step_weights[i] = weights

        best_next, best_val = min(step_results.items(), key=lambda x: x[1])
        selected_models.append(best_next)
        rmse_progression.append(best_val.item())
        model_progression.append(selected_models.copy())
        weight_boa_step[len(selected_models)] = step_weights[best_next]

        print(f"➤ Added {best_next} | Current overall agg MSE: {best_val:.5f}")

    best_step_idx = rmse_progression.index(min(rmse_progression))
    # force exactly max_models models
    best_model_combo = model_progression[-1]  # last step is size = max_models
    best_rmse = rmse_progression[-1]

    return (
        best_model_combo,
        best_rmse,
        rmse_progression,
        model_progression,
        weight_boa_step,
    )

#----------------------------------------------------------
#                     Forecast function for ensemble
#--------------------------------------------------------


def forecast_ensemble(
    dat_test,
    length_study,
    begin_test,
    regmat_tensor_test,
    dependent_var_tensor_test,
    regmat0_test,
    days_test,
    country,
    wd,
    price_s_lags,
    fuel_lags,
    da_lag,
    reg_names,
    dependent_index,
    active_regressor,
    batch_size,
    num_columns,
    device,
    best_model_combo,
    study,
    num_epochs_init,
    num_epochs_all,
    forecast_tensor_test,
    actual_test,
    weight_boa,
    mask_in_out_red,
    mask_in_out_full,
    model_type,
    model_types,
    studies,
    block_size
):

    output_dim = dat_test.shape[1]
    agg_sq_all = torch.zeros((length_study, output_dim), device=device)
    agg_forecast = torch.zeros((length_study, output_dim), device=device)
    unstandardized_y_test_all = torch.zeros((length_study, output_dim), device=device)
    unstandardized_outputs_all = torch.zeros((length_study, output_dim), device=device)

    for idx, trial_num in enumerate(best_model_combo):
        if study is None:
            model_type_idx = trial_num // block_size
            trial_idx = trial_num % block_size
            model_type = model_types[model_type_idx]
            studyy = studies[model_type]
            specific_trial = studyy.trials[trial_idx]
        else:
            specific_trial = study.trials[trial_num]
        learning_rate_init = specific_trial.params["learning_rate_init"]
        D_init = specific_trial.params["D_init"]
        learning_rate_all = specific_trial.params["learning_rate_all"]
        D_all = specific_trial.params["D_all"]

        weight_decay_init = specific_trial.params["weight_decay_init"]
        weight_decay_all = specific_trial.params["weight_decay_all"]

        lambda_reg_init = specific_trial.params["lambda_reg_init"]
        lambda_reg_all = specific_trial.params["lambda_reg_all"]

        if model_type in {2, 3, 5, 6, 8}:
            number_neurons = specific_trial.params["number_neurons"]
        else:
            number_neurons = 0

        if model_type in {7, 8}:
            use_ols_weights = True
            alpha = specific_trial.params["alpha"]

        else:
            use_ols_weights = None
            alpha = None


        mask_in_mid = torch.ones(
            (number_neurons, num_columns), dtype=torch.float32, device=device
        )
        mask_in_mid[:, dependent_index] = 0
        mask_input_to_mid = mask_in_mid
        previous_weights_s = None
        for n in range(length_study):
            if n == 0:
                learning_rate = learning_rate_init
                num_epochs = num_epochs_init
                D = D_init
                weight_decay = weight_decay_init
                lambda_reg = lambda_reg_init
                # extract the OLS coeffiecients
                if model_type in {7, 8}:
                    coef_OLS_all = forecast_expert_ext(
                        dat=dat_test[
                            (
                                begin_test
                                + (dat_test.shape[0] - regmat0_test.shape[0])
                                - D
                                + n
                            ) : (
                                begin_test
                                + (dat_test.shape[0] - regmat0_test.shape[0])
                                + n
                                + 1
                            ),
                            :,
                        ],
                        days=days_test[
                            (
                                begin_test
                                + (dat_test.shape[0] - regmat0_test.shape[0])
                                - D
                                + n
                            ) : (
                                begin_test
                                + (dat_test.shape[0] - regmat0_test.shape[0])
                                + n
                                + 1
                            )
                        ],
                        country=country,
                        reg_names=reg_names,
                        wd=wd,
                        price_s_lags=price_s_lags,
                        fuel_lags=fuel_lags,
                        da_lag=da_lag,
                    )["coefficients"]
                    coef_OLS_ten = torch.tensor(
                        coef_OLS_all.values, dtype=torch.float32, device=device
                    )
                    # Initialize an empty tensor to store OLS coeffiencts
                    ols_tensor = torch.zeros(
                        (output_dim, num_columns), dtype=torch.float32, device=device
                    )
                    # Populate ols_tensor with ols coefficient
                    for row_idx, col_indices in active_regressor.items():
                        ols_tensor[row_idx, col_indices] = coef_OLS_ten[
                            row_idx, : len(col_indices)
                        ]
                else:
                    ols_tensor = None

            else:
                learning_rate = learning_rate_all
                num_epochs = num_epochs_all
                D = D_all
                weight_decay = weight_decay_all
                lambda_reg = lambda_reg_all

            mean_x = regmat_tensor_test[(begin_test - D + n) : (begin_test + n),].mean(
                dim=0, keepdim=True
            )

            std_x = regmat_tensor_test[(begin_test - D + n) : (begin_test + n),].std(
                dim=0, keepdim=True, unbiased=False
            )
            std_y = dependent_var_tensor_test[
                (begin_test - D + n) : (begin_test + n),
            ].std(dim=0, keepdim=True, unbiased=False)

            std_x[std_x == 0] = 1

            X_train = (
                regmat_tensor_test[(begin_test - D + n) : (begin_test + n),] - mean_x
            ) / std_x

            # Now standardize TEST data with the SAME mean and std
            X_test = ((regmat_tensor_test[(begin_test + n),] - mean_x) / std_x).reshape(
                1, -1
            )

            # Scaling and standardization dependent_var_tensor_test
            mean_y = dependent_var_tensor_test[
                (begin_test - D + n) : (begin_test + n),
            ].mean(dim=0, keepdim=True)

            std_y[std_y == 0] = 1

            y_train = (
                dependent_var_tensor_test[(begin_test - D + n) : (begin_test + n),]
                - mean_y
            ) / std_y

            # Now standardize TEST data with the SAME mean and std
            y_test = (
                (dependent_var_tensor_test[(begin_test + n),] - mean_y) / std_y
            ).reshape(1, -1)

            # use dataloader
            dataset_train = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                dataset_train, batch_size=batch_size, shuffle=False
            )
            dataset_test = TensorDataset(X_test, y_test)
            test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

            num_feature = regmat_tensor_test.shape[1]


            _, _, previous_weights_s, unstandardized_y_test, unstandardized_outputs = (
                train_and_evaluate_updated_weights(
                    train_loader,
                    test_loader,
                    num_feature,
                    previous_weights_s,
                    num_epochs,
                    learning_rate,
                    number_neurons,
                    std_y,
                    mean_y,
                    mask_input_to_mid,
                    use_ols_weights,
                    ols_tensor,
                    weight_decay,
                    alpha,
                    lambda_reg,
                    output_dim,
                    mask_in_out_red,
                    mask_in_out_full,
                    device,
                    model_type,
            )
            )

            unstandardized_y_test_all[n,] = unstandardized_y_test.squeeze(0)

            unstandardized_outputs_all[n,] = unstandardized_outputs.squeeze(0)

        forecast_tensor_test[:, :, idx] = unstandardized_outputs_all

    M = len(best_model_combo) - 1
    for s in range(output_dim):
        expert_preds_s = forecast_tensor_test[:, s, :]

        actuals_s = actual_test[:, s]


        wt, forecast, agg_sq = boa_fully_adaptive(
            M, expert_preds_s, actuals_s, length_study, device, w0=None
        )

        weight_boa[s] = wt
        # agregate MSE for all hours
        agg_sq_all[:, s] = agg_sq

        agg_forecast[:, s] = forecast


    agg_sq_mean = agg_sq_all.mean(dim=0)
    overall_agg_mean = agg_sq_all.mean((0, 1))

    # Return timing details along with results
    return [agg_sq_mean, overall_agg_mean, unstandardized_y_test_all, agg_forecast]

#%%-----------------------------------------------------
#                     Ablation study for hyperparameters
#-----------------------------------------------------

def evaluate_parameter_grid(
    parameter_name,
    parameter_grid,
    dat_test,
    begin_test,
    regmat_tensor_test,
    dependent_var_tensor_test,
    regmat0_test,
    days_test,
    learning_rate_init,
    num_epochs_init,
    D_init,
    learning_rate_all,
    num_epochs_all,
    D_all,
    number_neurons,
    use_ols_weights,
    weight_decay_init,
    weight_decay_all,
    alpha,
    lambda_reg_init,
    lambda_reg_all,
    length_study,
    dependent_index,
    active_regressor,
    mask_in_out_red,
    mask_in_out_full,
    batch_size,
    wd,
    price_s_lags,
    fuel_lags,
    da_lag,
    num_columns,
    reg_names,
    country,
    device,
    model_type,
):



    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------
    out_dir = f"../Figures/Plots"
    os.makedirs(out_dir, exist_ok=True)

    results = []

    # --------------------------------------------------------------
    # Base arguments for forecast_MLP_rolling
    # --------------------------------------------------------------
    forecast_kwargs = {
        "dat_eval": dat_test,
        "begin_eval": begin_test,
        "regmat_tensor_eval": regmat_tensor_test,
        "dependent_var_tensor_eval": dependent_var_tensor_test,
        "regmat0_eval": regmat0_test,
        "days_eval": days_test,
        "learning_rate_init": learning_rate_init,
        "num_epochs_init": num_epochs_init,
        "D_init": D_init,
        "learning_rate_all": learning_rate_all,
        "num_epochs_all": num_epochs_all,
        "D_all": D_all,
        "number_neurons": number_neurons,
        "use_ols_weights": use_ols_weights,
        "weight_decay_init": weight_decay_init,
        "weight_decay_all": weight_decay_all,
        "alpha": alpha,
        "lambda_reg_init": lambda_reg_init,
        "lambda_reg_all": lambda_reg_all,
        "length_study": length_study,
        "dependent_index": dependent_index,
        "active_regressor": active_regressor,
        "mask_in_out_red": mask_in_out_red,
        "mask_in_out_full": mask_in_out_full,
        "batch_size": batch_size,
        "wd": wd,
        "price_s_lags": price_s_lags,
        "fuel_lags": fuel_lags,
        "da_lag": da_lag,
        "num_columns": num_columns,
        "reg_names": reg_names,
        "country": country,
        "device": device,
        "model_type": model_type,
    }

    # --------------------------------------------------------------
    # Check parameter name
    # --------------------------------------------------------------
    if parameter_name not in forecast_kwargs:
        raise ValueError(
            f"{parameter_name} is not a valid parameter. "
            f"Available parameters are:\n{list(forecast_kwargs.keys())}"
        )

    # --------------------------------------------------------------
    # Run parameter grid
    # --------------------------------------------------------------
    for parameter_value in parameter_grid:

        print(
            f"\nRunning {parameter_name} = {parameter_value}"
        )

        # Replace only the parameter being tested
        forecast_kwargs_current = forecast_kwargs.copy()
        forecast_kwargs_current[parameter_name] = parameter_value

        t0 = time.perf_counter()

        (
            overall_agg_mae,
            unstandardized_y_test,
            unstandardized_outputs_test,
        ) = forecast_MLP_rolling(
            **forecast_kwargs_current
        )

        wall_time = time.perf_counter() - t0

        # Convert MAE safely to float
        if torch.is_tensor(overall_agg_mae):
            overall_val = float(
                overall_agg_mae.detach().cpu().item()
            )
        else:
            overall_val = float(overall_agg_mae)

        results.append(
            {
                parameter_name: parameter_value,
                "overall_agg_mae": overall_val,
                "wall_time_sec": wall_time,
            }
        )

        print(
            f"{parameter_name} = {parameter_value} | "
            f"overall_agg_mae = {overall_val:8.4f} | "
            f"wall_time = {wall_time:8.2f} sec"
        )

    # --------------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------------
    df = (
        pd.DataFrame(results)
        .sort_values(parameter_name)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # Pareto front
    # --------------------------------------------------------------
    pts = df[
        ["wall_time_sec", "overall_agg_mae"]
    ].to_numpy()

    is_pareto = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):

        if not is_pareto[i]:
            continue

        dominated_by_other = (
            np.all(pts <= pts[i], axis=1)
            & np.any(pts < pts[i], axis=1)
        )

        dominated_by_other[i] = False

        if np.any(dominated_by_other):
            is_pareto[i] = False

    df["pareto"] = is_pareto

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------
    df.to_csv(
        f"{out_dir}/{parameter_name}_grid_results.csv",
        index=False,
    )



    return df

#------------------------------------------------------------
#              Calculate RMSE and MAE for all models
#------------------------------------------------------------

model_names = [
    "true",
    "RLin",
    "MLP",
    "MLP with RLin",
    "FLin",
    "MLP with FLin",
    "RLin with OLS",
    "MLP with RLin and OLS",
    "RLin (BOA)",
    "MLP (BOA)",
    "MLP with RLin (BOA)",
    "FLin (BOA)",
    "MLP with FLin (BOA)",
    "RLin with OLS (BOA)",
    "MLP with RLin and OLS (BOA)",
    "BOA all",
    "LEAR",
    "DNN",
    "GAM Online"
]

def calculate_metrics(country):





    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    base_dir = Path("..")

    online_dir = base_dir / "Online" / country
    benchmark_dir = base_dir / "Benchmark"
    data_dir = base_dir / "Data"

    # --------------------------------------------------
    # Load best forecasts
    # --------------------------------------------------

    models = [1, 2, 3, 4, 5, 7, 8]

    best_forecasts = []

    for model in models:

        path = (
            online_dir
            / f"Model{model}"
            / "unstandarized_forecast_best.pkl"
        )

        best_forecasts.append(
            np.asarray(joblib.load(path))
        )

    # --------------------------------------------------
    # Load true values
    # --------------------------------------------------

    true_path = (
        online_dir
        / "Model1"
        / "unstandardized_actual_best.pkl"
    )

    true_values = np.asarray(joblib.load(true_path))

    # --------------------------------------------------
    # Load BOA ensemble forecasts
    # --------------------------------------------------

    ensemble_forecasts = []

    for model in models:

        path = (
            online_dir
            / f"Model{model}"
            / "unstandarized_forecast_ensemble.pkl"
        )

        ensemble_forecasts.append(
            np.asarray(joblib.load(path))
        )

    # BOA using all models
    boa_all_path = (
        online_dir
        / "BOA_all"
        / "unstandarized_forecast_ensemble.pkl"
    )

    boa_all = np.asarray(joblib.load(boa_all_path))

    # --------------------------------------------------
    # Load benchmark forecasts
    # --------------------------------------------------

    forecast_lear = pd.read_csv(
        benchmark_dir
        / country
        / "experimental_files"
        / "LEAR_forecast_datmy_data_YT2_CW728.csv",
        index_col=False,
    )

    forecast_lear = forecast_lear.iloc[:, 1:].to_numpy()


    forecast_dnn = pd.read_csv(
        benchmark_dir
        / country
        / "experimental_files"
        / "DNN_forecast_nl2_datmy_data_YT2_SFH0_CW2_1.csv",
        index_col=False,
    )

    forecast_dnn = forecast_dnn.iloc[:, 1:].to_numpy()


    forecast_gam = pd.read_csv(
        benchmark_dir
        / f"{country}/forecast_results.csv"
    ).to_numpy()

    # --------------------------------------------------
    # Combine all forecasts
    # --------------------------------------------------

    length_study = 2 * 365
    output_dim = 24

    forecast_all = np.zeros(
        (length_study, output_dim, 19)
    )

    # True values
    forecast_all[:, :, 0] = true_values

    # Best individual models
    for i, forecast in enumerate(best_forecasts):
        forecast_all[:, :, i + 1] = forecast

    # BOA versions
    for i, forecast in enumerate(ensemble_forecasts):
        forecast_all[:, :, i + 8] = forecast

    # BOA all
    forecast_all[:, :, 15] = boa_all

    # Benchmarks
    forecast_all[:, :, 16] = forecast_lear
    forecast_all[:, :, 17] = forecast_dnn
    forecast_all[:, :, 18] = forecast_gam

    # --------------------------------------------------
    # RMSE and MAE
    # --------------------------------------------------

    errors = (
        forecast_all[:, :, 1:]
        - forecast_all[:, :, 0, None]
    )

    rmse = np.sqrt(
        np.mean(errors**2, axis=(0, 1))
    )

    mae = np.mean(
        np.abs(errors),
        axis=(0, 1)
    )

    # --------------------------------------------------
    # Create naive forecast for rMAE
    # --------------------------------------------------

    data = pd.read_csv(
        data_dir / f"{country}.csv"
    )

    time_utc = pd.to_datetime(
        data["time_utc"],
        utc=True,
        format="%Y-%m-%d %H:%M:%S"
    )

    local_time_zone = "CET"

    time_lt = time_utc.dt.tz_convert(
        local_time_zone
    )

    # Start/end times
    start_end_time_S = (
        time_lt.iloc[[0, -1]]
        .dt.tz_localize(None)
        .dt.tz_localize("UTC")
    )

    start_end_time_S_num = pd.to_numeric(
        start_end_time_S
    )

    time_S_numeric = np.arange(
        start=start_end_time_S_num.iloc[0],
        stop=(
            start_end_time_S_num.iloc[1]
            + 24 * 60 * 60 * 10**9 / output_dim
        ),
        step=24 * 60 * 60 * 10**9 / output_dim,
    )

    time_S = pd.Series(
        pd.to_datetime(time_S_numeric, utc=True)
    )

    dates_S = pd.Series(
        time_S.dt.date.unique()
    )

    # DST transformation
    data_array = DST_trafo(
        X=data.iloc[:, 1:],
        Xtime=time_utc,
        tz=local_time_zone
    )

    # --------------------------------------------------
    # Regression matrix
    # --------------------------------------------------

    reg_names = data.columns[1:]

    wd = [1, 6, 7]
    price_s_lags = [1, 2, 7]
    da_lag = [0]
    fuel_lags = [2]

    days_test = pd.to_datetime(dates_S)
    dat_test = data_array

    regmat_results = reg_matrix(
    dat_test,
    days_test,
    country,
    wd,
    reg_names,
    fuel_lags,
    price_s_lags,
    da_lag,
)

    regmat_test = regmat_results[0]
    columns_s = regmat_results[1]


    # Remove NA observations
    regmat0_test = regmat_test.dropna()

    # Dependent-variable columns
    dependent_index = [
        s * columns_s
        for s in range(output_dim)
    ]

    dependent_var_test = (
        regmat0_test.iloc[:, dependent_index]
    )

    actual_test = dependent_var_test.iloc[
        -length_study:
    ].to_numpy()

    # First observation of test period
    begin_test = (
        regmat0_test.shape[0]
        - length_study
    )

    # --------------------------------------------------
    # Naive forecast
    # --------------------------------------------------

    naive_forecast = np.zeros(
        (length_study, output_dim)
    )

    for n in range(length_study):

        dat = dat_test[
            : begin_test + n + 1,
            :,
            0
        ]

        days = days_test[
            : begin_test + n + 1
        ]

        weekdays_num = (
            days.iloc[-1].weekday() + 1
        )

        for s in range(output_dim):

            if weekdays_num in wd:

                naive_forecast[n, s] = dat[-2, s]

            else:

                naive_forecast[n, s] = dat[-8, s]

    # --------------------------------------------------
    # Naive MAE
    # --------------------------------------------------

    mae_naive = np.mean(
        np.abs(
            naive_forecast
            - actual_test
        )
    )

    # --------------------------------------------------
    # Relative MAE
    # --------------------------------------------------

    rmae = (
        mae / mae_naive
    ) * 100

    return rmse, mae, rmae, errors


def normalize_values(values):
    """Normalize values to range [0, 1]."""
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)

def value_to_color(norm_value):
    """
    Convert normalized value to RGB color.
    0 (min) -> light blue, 0.5 -> white, 1 (max) -> light red
    """
    norm_value = float(np.clip(norm_value, 0.0, 1.0))
    if norm_value <= 0.5:
        # Light blue to white
        r = int(255 * (1 - (0.5 - norm_value) * 0.6))
        g = int(255 * (1 - (0.5 - norm_value) * 0.6))
        b = 255
    else:
        # White to light red
        r = 255
        g = int(255 * (1 - (norm_value - 0.5) * 0.6))
        b = int(255 * (1 - (norm_value - 0.5) * 0.6))

    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"

def colored_cell(value, norm_value, is_best=False, is_worst=False, as_percent=False):
    # Keep cell color unchanged (same mapping for all values)
    color = value_to_color(norm_value)

    txt = f"{value:.2f}\\%" if as_percent else f"{value:.2f}"
    if is_best:  # Only bold the best (minimum) values
        txt = f"\\textbf{{{txt}}}"

    return f"\\cellcolor{color}{txt}"

def fmt_min_colored(val, col_min, norm_val, eps=1e-12):
    """Format value with bold text if minimum and add cell color."""
    color = value_to_color(norm_val)
    is_bold = abs(val - col_min) <= eps
    txt = f"{val:.2f}"
    if is_bold:
        txt = f"\\textbf{{{txt}}}"
    return f"\\cellcolor{color}{txt}"
#---------------------------------------------------------
#                        ADF Test
#----------------------------------------------------------
def loss_differential(error_a, error_b, loss, power=1):
    error_a = np.asarray(error_a)
    error_b = np.asarray(error_b)

    if loss == "L1":
        loss_a = (np.abs(error_a) ** power).sum(axis=1) ** (1 / power)
        loss_b = (np.abs(error_b) ** power).sum(axis=1) ** (1 / power)

    elif loss == "L2":
        loss_a = np.sum(error_a**2, axis=1)
        loss_b = np.sum(error_b**2, axis=1)

    else:
        raise ValueError("loss must be 'L1' or 'L2'")

    delta = loss_a - loss_b
    delta = delta[~np.isnan(delta)]

    return delta


def adf_test_loss_diff(error_a, error_b, loss, power=1):
    delta = loss_differential(error_a, error_b, loss=loss, power=power)

    #adf_result = adfuller(delta, autolag="AIC")
    adf_result = adfuller(delta,regression='c', maxlag=7)

    return {
        "ADF_stat": adf_result[0],
        "ADF_pval": adf_result[1],
        "stationary": adf_result[1] < 0.05,
    }

#---------------------------------------------------------
#              Plot ADF Stationarity
#---------------------------------------------------------

def plot_adf_stationarity(model_names,country, loss, errors):

    import os
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # --------------------------------------------------
    # Model names without True values
    # --------------------------------------------------
    model_names_wo_true = [
        model_names[i + 1]
        for i in range(len(model_names) - 1)
    ]

    # --------------------------------------------------
    # ADF result matrices
    # --------------------------------------------------
    adf_pvals = pd.DataFrame(
        index=model_names_wo_true,
        columns=model_names_wo_true,
        dtype=float,
    )

    adf_stats = pd.DataFrame(
        index=model_names_wo_true,
        columns=model_names_wo_true,
        dtype=float,
    )

    adf_stationary = pd.DataFrame(
        index=model_names_wo_true,
        columns=model_names_wo_true,
        dtype=object,
    )

    # --------------------------------------------------
    # Run pairwise ADF tests
    # --------------------------------------------------
    for i_a, mod_a in enumerate(model_names_wo_true):

        for i_b, mod_b in enumerate(model_names_wo_true):

            if mod_a == mod_b:

                adf_pvals.at[mod_a, mod_b] = np.nan
                adf_stats.at[mod_a, mod_b] = np.nan
                adf_stationary.at[mod_a, mod_b] = np.nan

            else:

                result = adf_test_loss_diff(
                    errors[..., i_a],
                    errors[..., i_b],
                    loss=loss,
                    power=1,
                )

                adf_pvals.at[mod_a, mod_b] = result["ADF_pval"]
                adf_stats.at[mod_a, mod_b] = result["ADF_stat"]
                adf_stationary.at[mod_a, mod_b] = result["stationary"]

    # --------------------------------------------------
    # Convert True / False to numeric
    # --------------------------------------------------
    adf_plot_matrix = adf_stationary.replace(
        {
            True: 1,
            False: 0,
        }
    ).astype(float)

    plot_model_names = adf_plot_matrix.index.tolist()

    # --------------------------------------------------
    # Cell text: p-values
    # --------------------------------------------------
    cell_text = np.empty(
        adf_pvals.shape,
        dtype=object,
    )

    for i in range(adf_pvals.shape[0]):

        for j in range(adf_pvals.shape[1]):

            pval = adf_pvals.iloc[i, j]

            if pd.isna(pval):

                cell_text[i, j] = ""

            else:

                if pval < 1e-20:

                    cell_text[i, j] = "&lt;10<sup>-20</sup>"

                else:

                    mantissa, exponent = f"{pval:.1e}".split("e")

                    mantissa = float(mantissa)
                    exponent = int(exponent)

                    if mantissa > 1:
                        exponent += 1

                    cell_text[i, j] = (
                        f"10<sup>{exponent}</sup>"
                    )

    # --------------------------------------------------
    # Colors
    # --------------------------------------------------
    blue_yellow_scale = [
        [0.0, "#FDE725"],
        [1.0, "#2C7BB6"],
    ]

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    fig = go.Figure()

    # Main heatmap
    fig.add_trace(
        go.Heatmap(
            z=adf_plot_matrix.values,
            x=plot_model_names,
            y=plot_model_names,
            zmin=0,
            zmax=1,
            colorscale=blue_yellow_scale,
            showscale=False,
            text=cell_text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate=(
                "Model A: %{y}<br>"
                "Model B: %{x}<br>"
                "ADF p-value: %{text}"
                "<extra></extra>"
            ),
        )
    )

    # Stationary legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=18,
                color="#2C7BB6",
                symbol="square",
            ),
            name="Stationary",
            showlegend=True,
        )
    )

    # Non-stationary legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=18,
                color="#FDE725",
                symbol="square",
            ),
            name="Non-Stationary",
            showlegend=True,
        )
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_layout(
        xaxis_title="Model B",
        yaxis_title="Model A",
        width=900,
        height=850,
        font=dict(size=12),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=-0.5,
            font=dict(size=12),
        ),

        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10,
        ),
    )

    # --------------------------------------------------
    # Axes
    # --------------------------------------------------
    fig.update_xaxes(
        tickangle=45,
        side="top",
        tickfont=dict(size=11),
        title_font=dict(size=12),
    )

    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=11),
        title_font=dict(size=12),
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs("Plots", exist_ok=True)

    output_file = (
        f"Plots/adf_{country.lower()}_stationarity_"
        f"{loss}_pvalues.pdf"
    )

    fig.write_image(
        output_file,
        scale=4,
    )

    # --------------------------------------------------
    # Show
    # --------------------------------------------------
    fig.show()

    return fig

#---------------------------------------------------------
#                    DM Test
#---------------------------------------------------------


def dm_test(loss, error_a, error_b, hmax=1, L=7, small_sample=True):
    if loss == "L1":
        loss_a = (np.abs(error_a)).sum(1) 
        loss_b = (np.abs(error_b)).sum(1) 
    elif loss == "L2":
        loss_a = np.sum(error_a**2, axis=1)
        loss_b = np.sum(error_b**2, axis=1)     
    
    delta = loss_a - loss_b
    Tn = len(delta)

    # --- regression on constant with HAC SE ---
    X = np.ones((Tn, 1))
    res = sm.OLS(delta, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(L)})

    stat = float(res.tvalues[0])  # t-stat for intercept = mean(delta)
    # one-sided: A better than B => mean(delta) < 0
    p_val = float(t.cdf(stat, df=Tn - 1))

    # --- HLN small-sample correction (optional, like your code) ---
    if small_sample:
        k = np.sqrt((Tn + 1 - 2 * hmax + (hmax / Tn) * (hmax - 1)) / Tn)
        stat *= k
        # p-value should match corrected stat
        p_val = float(t.cdf(stat, df=Tn - 1))

    return {"stat": stat, "p_val": p_val}

def plot_dm_heatmap(loss, errors, model_names, country):



    # --------------------------------------------------
    # Model names without True values
    # --------------------------------------------------
    model_names_wo_true = model_names[1:]

    # --------------------------------------------------
    # Empty result matrices
    # --------------------------------------------------
    dm_results_df = pd.DataFrame(
        index=model_names_wo_true,
        columns=model_names_wo_true,
        dtype=float,
    )

    dm_results = {
        "p_val": dm_results_df.copy(),
        "t_stat": dm_results_df.copy(),
    }

    # --------------------------------------------------
    # Pairwise DM tests
    # --------------------------------------------------
    for i_a, mod_a in enumerate(model_names_wo_true):

        for i_b, mod_b in enumerate(model_names_wo_true):

            if mod_a == mod_b:

                dm_results["p_val"].loc[mod_a, mod_b] = np.nan
                dm_results["t_stat"].loc[mod_a, mod_b] = np.nan

            else:

                dm = dm_test(
                    loss,
                    errors[..., i_a],
                    errors[..., i_b],
                )

                dm_results["p_val"].loc[mod_a, mod_b] = dm["p_val"]
                dm_results["t_stat"].loc[mod_a, mod_b] = dm["stat"]

    # --------------------------------------------------
    # P-value matrix
    # --------------------------------------------------
    p_val_matrix = dm_results["p_val"].astype(float)

    plot_model_names = p_val_matrix.index.tolist()

    # --------------------------------------------------
    # Create heatmap
    # --------------------------------------------------
    fig = go.Figure(
        data=go.Heatmap(
            z=p_val_matrix.values,
            x=plot_model_names,
            y=plot_model_names,

            zmin=0,
            zmax=0.1,

            colorscale="RdYlGn_r",

            colorbar=dict(
                title="P-Value"
            ),

            hovertemplate=(
                "Model A: %{y}<br>"
                "Model B: %{x}<br>"
                "<b>P-Value</b>: %{z:.4f}"
                "<extra></extra>"
            ),
        )
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_layout(
        xaxis_title="Model B",
        yaxis_title="Model A",

        width=750,
        height=700,

        font=dict(size=10),

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
    )

    # --------------------------------------------------
    # Axis formatting
    # --------------------------------------------------
    fig.update_xaxes(
        tickangle=45,
        side="top",
        tickfont=dict(size=10),
        title_font=dict(size=10),
    )

    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=10),
        title_font=dict(size=10),
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs("Plots", exist_ok=True)

    output_file = (
        f"Plots/heatmap_pval_{loss}_{country.lower()}.pdf"
    )

    fig.write_image(
        output_file,
        scale=4,
    )

    # --------------------------------------------------
    # Show
    # --------------------------------------------------
    fig.show()

    return fig, dm_results

#-----------------------------------------------------------
#                    Time vs MAE Pareto Plot
# -----------------------------------------------------------    

def plot_pareto_germany_spain(
    maes_germany,
    runtimes_germany,
    maes_spain,
    runtimes_spain,
    model_names,
):

    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # --------------------------------------------------
    # Figure sizing / typography
    # --------------------------------------------------
    FIG_WIDTH = 980
    FIG_HEIGHT = 520

    FONT_FAMILY = "DejaVu Sans"
    FONT_BASE = 16
    FONT_AXIS_TITLE = 18
    FONT_TICK = 14
    FONT_LEGEND = 13
    FONT_SUBPLOT = 18

    # --------------------------------------------------
    # Pareto function
    # --------------------------------------------------
    def compute_pareto(maes, runtimes, names):

        points = sorted(
            zip(runtimes, maes, names),
            key=lambda x: x[0],
        )

        pareto = []
        best_mae = float("inf")

        for rt, mae, name in points:

            if mae < best_mae:

                pareto.append(
                    (rt, mae, name)
                )

                best_mae = mae

        return points, pareto

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    model_list = list(model_names[1:])  # exclude "true"

    # --------------------------------------------------
    # Symbols
    # --------------------------------------------------
    symbol_pool = [
        "circle",
        "square",
        "diamond",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "hexagram",
        "star-square",
        "star-triangle-up",
        "star-triangle-down",
        "cross",
        "x",
        "hourglass",
    ]

    if len(model_list) > len(symbol_pool):

        raise ValueError(
            f"Not enough symbols: "
            f"{len(model_list)} models but only "
            f"{len(symbol_pool)} symbols."
        )

    symbol_by_model = {
        model: symbol_pool[i]
        for i, model in enumerate(model_list)
    }

    # ==================================================
    # Germany
    # ==================================================
    points_de, pareto_de = compute_pareto(
        maes_germany,
        runtimes_germany,
        model_list,
    )

    (
        all_rts_de,
        all_maes_de,
        all_labels_de,
    ) = zip(*points_de)

    (
        pareto_rts_de,
        pareto_maes_de,
        pareto_labels_de,
    ) = zip(*pareto_de)

    hover_all_de = [
        (
            f"<b>{name}</b><br>"
            f"Runtime: {rt:.1f}s<br>"
            f"MAE: {mae:.3f}"
        )
        for rt, mae, name in points_de
    ]

    hover_pareto_de = [
        (
            f"<b>{name}</b><br>"
            f"Runtime: {rt:.1f}s<br>"
            f"MAE: {mae:.3f}"
        )
        for rt, mae, name in pareto_de
    ]

    all_symbols_de = [
        symbol_by_model[name]
        for name in all_labels_de
    ]

    pareto_symbols_de = [
        symbol_by_model[name]
        for name in pareto_labels_de
    ]

    # ==================================================
    # Spain
    # ==================================================
    points_es, pareto_es = compute_pareto(
        maes_spain,
        runtimes_spain,
        model_list,
    )

    (
        all_rts_es,
        all_maes_es,
        all_labels_es,
    ) = zip(*points_es)

    (
        pareto_rts_es,
        pareto_maes_es,
        pareto_labels_es,
    ) = zip(*pareto_es)

    hover_all_es = [
        (
            f"<b>{name}</b><br>"
            f"Runtime: {rt:.1f}s<br>"
            f"MAE: {mae:.3f}"
        )
        for rt, mae, name in points_es
    ]

    hover_pareto_es = [
        (
            f"<b>{name}</b><br>"
            f"Runtime: {rt:.1f}s<br>"
            f"MAE: {mae:.3f}"
        )
        for rt, mae, name in pareto_es
    ]

    all_symbols_es = [
        symbol_by_model[name]
        for name in all_labels_es
    ]

    pareto_symbols_es = [
        symbol_by_model[name]
        for name in pareto_labels_es
    ]

    # ==================================================
    # Figure
    # ==================================================
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "German-Luxembourg",
            "Spain",
        ),
        shared_yaxes=False,
        horizontal_spacing=0.12,
    )

    # --------------------------------------------------
    # Germany: all models
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=all_rts_de,
            y=all_maes_de,
            mode="markers",
            marker=dict(
                color="royalblue",
                size=6,
                symbol=all_symbols_de,
                opacity=0.55,
                line=dict(
                    width=0.7,
                    color="midnightblue",
                ),
            ),
            hovertext=hover_all_de,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Germany: Pareto
    fig.add_trace(
        go.Scatter(
            x=pareto_rts_de,
            y=pareto_maes_de,
            mode="lines+markers",
            line=dict(
                color="crimson",
                width=1.7,
            ),
            marker=dict(
                color="crimson",
                size=8,
                symbol=pareto_symbols_de,
                opacity=0.72,
                line=dict(
                    width=0.7,
                    color="darkred",
                ),
            ),
            hovertext=hover_pareto_de,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------
    # Spain: all models
    # --------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=all_rts_es,
            y=all_maes_es,
            mode="markers",
            marker=dict(
                color="royalblue",
                size=6,
                symbol=all_symbols_es,
                opacity=0.55,
                line=dict(
                    width=0.7,
                    color="midnightblue",
                ),
            ),
            hovertext=hover_all_es,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Spain: Pareto
    fig.add_trace(
        go.Scatter(
            x=pareto_rts_es,
            y=pareto_maes_es,
            mode="lines+markers",
            line=dict(
                color="crimson",
                width=1.7,
            ),
            marker=dict(
                color="crimson",
                size=8,
                symbol=pareto_symbols_es,
                opacity=0.72,
                line=dict(
                    width=0.7,
                    color="darkred",
                ),
            ),
            hovertext=hover_pareto_es,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------
    for model in model_list:

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    symbol=symbol_by_model[model],
                    size=8,
                    color="black",
                ),
                name=model,
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # --------------------------------------------------
    # Axes
    # --------------------------------------------------
    fig.update_xaxes(
        title="Runtime (Seconds, log scale)",
        type="log",
        tickmode="array",
        tickvals=[
            1,
            10,
            100,
            1000,
            10000,
        ],
        ticktext=[
            "1",
            "10",
            "100",
            "1000",
            "10000",
        ],
        title_font=dict(
            family=FONT_FAMILY,
            size=FONT_AXIS_TITLE,
        ),
        tickfont=dict(
            family=FONT_FAMILY,
            size=FONT_TICK,
        ),
    )

    fig.update_yaxes(
        title="MAE",
        title_font=dict(
            family=FONT_FAMILY,
            size=FONT_AXIS_TITLE,
        ),
        tickfont=dict(
            family=FONT_FAMILY,
            size=FONT_TICK,
        ),
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_layout(
        template="plotly_white",

        font=dict(
            family=FONT_FAMILY,
            size=FONT_BASE,
            color="black",
        ),

        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.20,
            yanchor="bottom",

            font=dict(
                family=FONT_FAMILY,
                size=FONT_LEGEND,
            ),

            entrywidthmode="fraction",
            entrywidth=0.24,
        ),

        margin=dict(
            l=45,
            r=25,
            t=75,
            b=70,
        ),

        width=FIG_WIDTH,
        height=FIG_HEIGHT,
    )

    # --------------------------------------------------
    # Subplot titles
    # --------------------------------------------------
    for ann in fig.layout.annotations:

        ann.font = dict(
            family=FONT_FAMILY,
            size=FONT_SUBPLOT,
            color="black",
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs("Plots", exist_ok=True)

    fig.write_image(
        "Plots/pareto_germany_spain.pdf",
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        scale=1,
    )

    # --------------------------------------------------
    # Show
    # --------------------------------------------------
    fig.show()

    return fig, pareto_de, pareto_es


