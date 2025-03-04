import numpy as np
from sfms import SFMSRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from activ_function_elusplus2L import Elusplus2L
from sklearn.metrics import make_scorer

n_features = 10
n_prediction_steps = 3

scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

param_grid = {
    "n_hidden": [4, 8],
    "n_shared": [0, 1, 2],
    "n_specific": [0, 1, 2],
    "dropout": [0.2, 0.4, 0.6],
}

method = RandomizedSearchCV(
    SFMSRegressor(
        n_features=n_features,
        learning_rate=0.0001,
        activation=Elusplus2L(),
        use_reduce_lr_on_plateau=True,
        early_stopping_patience=50,
        batch_size=50,
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
    error_score="raise",
)

X = np.random.rand(1000, n_features)
y = np.random.rand(1000, n_prediction_steps)

X_train = X[:600]
y_train = y[:600]
X_val = X[600:800]
y_val = y[600:800]
X_test = X[800:]
y_test = y[800:]

### Fit estimator measuring time
##
#
method.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)

### Predict
##
#
y_pred = method.predict(X_test)

### Evaluate
##
#
print("")
rmses = []
for pred_step in range(n_prediction_steps):
    rmses.append(root_mean_squared_error(y_test[:, pred_step], y_pred[:, pred_step]))
print("RMSE: ", np.mean(rmses))
