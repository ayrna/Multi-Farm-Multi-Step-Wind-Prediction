import numpy as np
from mfms import MFMSRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from activ_function_elusplus2L import Elusplus2L
from sklearn.metrics import make_scorer


def multifarm_rmse(y_true, y_pred, Q, n_prediction_steps):
    return root_mean_squared_error(
        y_true.reshape(-1, Q * n_prediction_steps), y_pred.reshape(-1, Q * n_prediction_steps)
    )


Q = 3
n_features_per_source = 10
n_prediction_steps = 3

scorer = make_scorer(multifarm_rmse, greater_is_better=False, Q=Q, n_prediction_steps=n_prediction_steps)

param_grid = {
    "n_hidden": [4, 8],
    "n_shared": [0, 1, 2],
    "n_specific": [0, 1, 2],
    "dropout": [0.2, 0.4, 0.6],
}

method = RandomizedSearchCV(
    MFMSRegressor(
        n_features=n_features_per_source * Q,
        learning_rate=0.0001,
        activation=Elusplus2L(),
        use_reduce_lr_on_plateau=True,
        early_stopping_patience=50,
        batch_size=200,
        max_epochs=50,
        device="cpu",
        verbose=0,
        n_jobs=1,
    ),
    scoring=scorer,
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    n_jobs=1,
    verbose=3,
)

multisource_X = np.empty((1000, n_features_per_source * Q))
multisource_y = np.empty((1000, Q, n_prediction_steps))
for source in range(Q):
    X = np.random.rand(1000, n_features_per_source)
    y = np.random.rand(1000, n_prediction_steps)
    multisource_X[:, source * n_features_per_source : (source + 1) * n_features_per_source] = X
    multisource_y[:, source, :] = y

multisource_X_train = multisource_X[:600]
multisource_y_train = multisource_y[:600]
multisource_X_val = multisource_X[600:800]
multisource_y_val = multisource_y[600:800]
multisource_X_test = multisource_X[800:]
multisource_y_test = multisource_y[800:]

### Fit estimator measuring time
##
#
method.fit(X=multisource_X_train, y=multisource_y_train, X_val=multisource_X_val, y_val=multisource_y_val)

### Predict
##
#
y_pred = method.predict(multisource_X_test)

### Evaluate
##
#
print("")
print("Test RMSE =", multifarm_rmse(multisource_y_test, y_pred, Q, n_prediction_steps))
