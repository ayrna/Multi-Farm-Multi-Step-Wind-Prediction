import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import root_mean_squared_error
from copy import deepcopy


class SISONetRegressor(BaseEstimator, ClassifierMixin):
    """
    Implementation of the SIngle-SOurce deep neural Network (SISONet) model for multi-step regression.
    """

    def __init__(
        self,
        *,
        n_features=10,
        n_hidden=32,
        n_shared=0,
        n_specific=0,
        dropout=0.0,
        activation=nn.ReLU(),
        device="cpu",
        max_epochs=1000,
        learning_rate=1e-3,
        use_reduce_lr_on_plateau=True,
        early_stopping_patience=None,
        verbose=0,
        batch_size=128,
        n_jobs=1,
    ):

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_shared = n_shared
        self.n_specific = n_specific
        self.activation = activation
        self.dropout = dropout
        self.device = device
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.use_reduce_lr_on_plateau = use_reduce_lr_on_plateau
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.best_params_ = {}
        self.reduce_lr_on_plateau = None

        self._estimator_type = "regressor"

    def _initialize(self):

        self.model = self._setup_model()
        self._setup_losses()

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        if self.use_reduce_lr_on_plateau:
            self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                Adam(self.model.parameters(), lr=self.learning_rate),
                mode="min",
                factor=0.1,
                patience=10,
                threshold=0.001,
                threshold_mode="rel",
                cooldown=0,
                min_lr=0,
                eps=1e-08,
                verbose="deprecated",
            )

        if hasattr(self.activation, "_initialize"):
            self.activation._initialize(self.device)

        self.loss_history_train = []
        self.loss_history_val = []

    def _setup_model(self):
        model = SISONet(
            n_input=self.n_features,
            n_hidden=self.n_hidden,
            n_shared=self.n_shared,
            n_specific=self.n_specific,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self.device)
        return model

    def _setup_losses(self):
        self.loss = nn.MSELoss()

    def _forward(self, X):
        output = self.model(X)
        return output

    def set_train_mode(self, mode):
        self.model.train(mode)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._initialize()

        X_train = torch.tensor(X_train, dtype=torch.float).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float).to(self.device)

        N = X_train.shape[0]

        burnt_patience = 0
        self._initial_model = deepcopy(self.model.state_dict())
        self._best_model = deepcopy(self.model.state_dict())
        best_val_rmse = np.inf
        best_n_epochs = self.max_epochs
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            n_batches = (N // self.batch_size) + 1
            for j in range(n_batches):
                self.set_train_mode(True)
                self.optimizer.zero_grad()
                X_train_batch = X_train[(j * self.batch_size) : ((j + 1) * self.batch_size)]
                y_train_batch = y_train[(j * self.batch_size) : ((j + 1) * self.batch_size), :]

                pred_train_batch = self._forward(X_train_batch)

                batch_loss = self.loss(pred_train_batch, y_train_batch)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                epoch_loss = epoch_loss / (N // self.batch_size)

                pred_train = self._forward(X_train)
                pred_val = self._forward(X_val)

                train_loss = self.loss(pred_train, y_train).item()
                self.loss_history_train.append(train_loss)
                val_loss = self.loss(pred_val, y_val).item()
                self.loss_history_val.append(val_loss)

                val_rmse = root_mean_squared_error(y_val.cpu().numpy(), pred_val.cpu().numpy())
                if epoch == 0:
                    best_val_rmse = val_rmse
                else:
                    if val_rmse < best_val_rmse:
                        self._best_model = deepcopy(self.model.state_dict())
                        best_val_rmse = val_rmse
                        burnt_patience = 0
                    else:
                        burnt_patience += 1

                # Early stopping
                if self.early_stopping_patience is not None:
                    if burnt_patience >= self.early_stopping_patience:
                        best_n_epochs = epoch - burnt_patience
                        break

                # Reduce LR on plateau
                if self.use_reduce_lr_on_plateau is not None:
                    self.reduce_lr_on_plateau.step(val_rmse)

                if self.verbose > 0:
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} | train loss: {round(epoch_loss, 4)} | val loss: {round(val_loss, 4)}"
                    )

                if self.verbose > 1 and epoch % 10 == 0:

                    self.set_train_mode(False)

                    if self.use_reduce_lr_on_plateau is not None:
                        print("* Current LR:", self.reduce_lr_on_plateau.get_last_lr())

                    print("\n== Validation ========================")
                    if self.early_stopping_patience is not None:
                        print(f"* Best val RMSE: {best_val_rmse}")
                        print(f"* Burnt patience: {burnt_patience}")
                    print("RMSE:", val_rmse)
                    print(
                        "RSME → 6h:",
                        root_mean_squared_error(
                            y_val[:, 0].cpu().numpy(),
                            pred_val[:, 0].cpu().numpy(),
                        ),
                    )
                    print(
                        "RSME → 12h:",
                        root_mean_squared_error(
                            y_val[:, 1].cpu().numpy(),
                            pred_val[:, 1].cpu().numpy(),
                        ),
                    )
                    print(
                        "RSME → 24h:",
                        root_mean_squared_error(
                            y_val[:, 2].cpu().numpy(),
                            pred_val[:, 2].cpu().numpy(),
                        ),
                    )
                    print("========================================\n")

        self.model.load_state_dict(self._best_model, strict=True, assign=True)

        with torch.no_grad():
            # retrain on concatenated train and val sets with the best n_epochs found
            X_train_val = torch.cat((X_train, X_val), dim=0)
            y_train_val = torch.cat((y_train, y_val), dim=0)
            # self.refit(X_train_val, y_train_val, n_epochs=best_n_epochs)

        return self

    def refit(self, X_train, y_train, n_epochs):
        self._initialize()

        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float).to(self.device)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)

        N = X_train.shape[0]

        self._best_model = deepcopy(self.model.state_dict())
        for _ in range(n_epochs):
            epoch_loss = 0
            n_batches = (N // self.batch_size) + 1
            for j in range(n_batches):
                self.set_train_mode(True)
                self.optimizer.zero_grad()
                X_train_batch = X_train[(j * self.batch_size) : ((j + 1) * self.batch_size)]
                y_train_batch = y_train[(j * self.batch_size) : ((j + 1) * self.batch_size), :]

                pred_train_batch = self._forward(X_train_batch)

                batch_loss = self.loss(pred_train_batch, y_train_batch)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

        self.model.load_state_dict(self._best_model, strict=True, assign=True)

        return self

    def predict(self, X):
        with torch.no_grad():
            self.set_train_mode(False)
            X = torch.tensor(X, dtype=torch.float).to(self.device)
            return self._forward(X).cpu().numpy()


class SISONet(nn.Module):
    """
    Implementation of the network for the SISONet model.
    """

    def __init__(self, n_input, n_hidden, n_shared, n_specific, activation, dropout):
        super(SISONet, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_shared = n_shared
        self.n_specific = n_specific
        self.activation = activation
        self.dropout = dropout

        # Weights and biases for the layers
        self.fc_inc1 = nn.Linear(n_input, 2 * n_hidden)
        self.fc_c1c2 = nn.Linear(2 * n_hidden, n_hidden // 2)
        self.fc_c2c3 = nn.Linear(n_hidden // 2, n_hidden)

        if n_shared > 0:
            self.shared_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_shared)])

        self.fc_cf1s1 = nn.Linear(n_hidden, n_hidden)
        if n_specific > 0:
            self.farm1_specific_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_specific)])
        self.fc_out_f1 = nn.Linear(n_hidden, 3)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # COMMON LAYERS
        c1 = self.activation(self.fc_inc1(x))
        c1 = self.dropout(c1)
        c2 = self.activation(self.fc_c1c2(c1))
        c2 = self.dropout(c2)
        c3 = self.activation(self.fc_c2c3(c2))
        c3 = self.dropout(c3)

        if self.n_shared > 0:
            for shared_layer in self.shared_layers:
                c3 = self.activation(shared_layer(c3))
                c3 = self.dropout(c3)

        # FARM 1
        cf1s1 = self.activation(self.fc_cf1s1(c3))
        cf1s1 = self.dropout(cf1s1)
        if self.n_specific > 0:
            for farm1_specific_layer in self.farm1_specific_layers:
                cf1s1 = self.activation(farm1_specific_layer(cf1s1))
                cf1s1 = self.dropout(cf1s1)
        output = self.fc_out_f1(cf1s1)

        return output
