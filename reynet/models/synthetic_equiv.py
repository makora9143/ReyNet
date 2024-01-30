import torch
from torch import nn
from torch import Tensor
from torch import optim
from pytorch_lightning import LightningModule

from ..layers.equiv.reynolds import ReyEquiv2to2
from ..layers.equiv.maron import MaronEquivNet2to2
from ..loss import CornerMSELoss


class ReyNetModel(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.model = ReyEquiv2to2(1,
                                  set_layers=self.args.set_layers,
                                  channel_layers=self.args.channel_layers)
        self.criterion = CornerMSELoss()
        self.save_hyperparameters()

    def forward(self, x: Tensor, shortcut: bool = False) -> Tensor:
        preds = self.model(x, shortcut)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        y_hat = self.model(x, shortcut=True)
        loss = self.criterion(y_hat, y)

        with torch.no_grad():
            y_pred = self.model(x)
            error = self.criterion(y_pred, y, corner=False)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_error", error, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        y_hat = self.model(x, shortcut=True)
        loss = self.criterion(y_hat, y)

        y_pred = self.model(x)
        error = self.criterion(y_pred, y, corner=False)
        self.log("val_loss", loss)
        self.log("val_error", error)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self.model(x)
        error = self.criterion(y_pred, y, corner=False)
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ReyNetModel")
        parser.add_argument('--set-layers', type=int, nargs='+', default=[128])
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[1])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser


class MaronModel(LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.model = MaronEquivNet2to2(
            1, self.args.channel_layers
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
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[80])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser


class MLPModel(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        phi = [nn.Linear(self.args.set_size * self.args.set_size, self.args.layers[0]), nn.ReLU()]
        for i in range(1, len(self.args.layers)):
            phi += [nn.Linear(self.args.layers[i - 1], self.args.layers[i]), nn.ReLU()]
        phi += [nn.Linear(self.args.layers[-1], self.args.set_size * self.args.set_size)]
        self.model = nn.Sequential(*phi)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        B, C, n, n = x.shape
        x = self.model(x.reshape(B, C, -1)).reshape(B, C, n, n)
        return x

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_pred = self(x)
        error = self.criterion(y_pred, y)
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MLPModel")
        parser.add_argument('--layers', type=int, nargs='+', default=[128])
        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser
