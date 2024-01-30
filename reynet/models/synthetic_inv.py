from torch import Tensor
from torch import nn
from torch import optim
import pytorch_lightning as pl

from ..layers.equiv.reynolds import ReyEquiv2to2
from ..layers.equiv.maron import MaronEquivNet2to2
from ..layers.pooling import DiagOffdiagSumpool


class ReyNetModel(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.model = nn.Sequential(
            ReyEquiv2to2(1, set_layers=self.args.set_layers, channel_layers=self.args.channel_layers),
            DiagOffdiagSumpool(),
            nn.Linear(2 * self.args.channel_layers[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.model(x)
        error = self.criterion(y_pred, y)
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ReyNetModel")
        parser.add_argument('--set-layers', type=int, nargs='+', default=[128])
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser


class MaronModel(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.model = nn.Sequential(
            MaronEquivNet2to2(1, self.args.channel_layers),
            DiagOffdiagSumpool(),
            nn.Linear(2 * self.args.channel_layers[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.model(x)
        error = self.criterion(y_pred, y)
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MaronModel")
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser


class MLPModel(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args


        self.model = nn.Sequential(
            ReyEquiv2to2(1, set_layers=set_layers, channel_layers=channel_layers),
            DiagOffdiagSumpool(),
            nn.Linear(2 * channel_layers[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.model(x)
        error = self.criterion(y_pred, y)
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MLPModel")
        parser.add_argument('--set-layers', type=int, nargs='+', default=[128])
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser