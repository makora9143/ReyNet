from torch import Tensor
from torch import nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics import Accuracy

from torch_geometric.nn import GIN, global_add_pool

from ..layers.equiv.sannai import ReyEquiv2to2
from ..layers.equiv.maron import MaronEquivNet2to2
from ..layers.pooling import DiagOffdiagSumpool
from ..datasets.graph import DECAY_RATES, LEARNING_RATES


class ReyNetModel(pl.LightningModule):
    def __init__(self, in_features, num_classes, args) -> None:
        super().__init__()
        self.args = args

        self.model = nn.Sequential(
            ReyEquiv2to2(in_features, set_layers=[128, 256, 512], channel_layers=[16, 32, 256]),
            DiagOffdiagSumpool(),
            nn.Linear(2 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        def initialize_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        self.model.apply(initialize_weights)

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATES[self.args.dataset])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 20,
            }
        }

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        x = x.float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        x, y = x.float(), y.unsqueeze(0)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_acc", self.metric(y_hat, y))
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x, y = x.float(), y.unsqueeze(0)
        y_hat = self.model(x)
        error = self.criterion(y_hat, y)
        self.log("test_acc", self.metric(y_hat, y))
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ReyNetModel")
        parser.add_argument('--set-layers', type=int, nargs='+', default=[128])
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser


class MaronModel(pl.LightningModule):
    def __init__(self, in_features, num_classes, args) -> None:
        super().__init__()
        self.args = args

        self.model = nn.Sequential(
            MaronEquivNet2to2(in_features, [16, 32, 256]),
            DiagOffdiagSumpool(),
            nn.Linear(2 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATES[self.args.dataset])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATES[self.args.dataset])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 20,
            }
        }

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        x = x.float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        x, y = x.float(), y.unsqueeze(0)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_acc", self.metric(y_hat, y))
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x, y = x.float(), y.unsqueeze(0)
        y_hat = self.model(x)
        error = self.criterion(y_hat, y)
        self.log("test_acc", self.metric(y_hat, y))
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ReyNetModel")
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser


class GINModel(pl.LightningModule):
    def __init__(self, in_features, num_classes, args) -> None:
        super().__init__()
        self.args = args

        self.gnn = GIN(in_features, self.args.hidden_channels, args.num_layers, jk='cat')
        self.classifier = nn.Sequential(
            nn.Linear(self.args.hidden_channels, 512)
        )
        # self.classifier = MLP([2 * self.args.hidden_channels, 512, 256, num_classes])

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x: Tensor, edge_index: Tensor, batch) -> Tensor:
        h = self.gnn(x, edge_index)
        h = global_add_pool(h, batch)
        h = self.classifier(h)
        return x

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATES[self.args.dataset])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATES[self.args.dataset])
        return [optimizer], [scheduler]

    def training_step(self, train_data, batch_idx) -> Tensor:
        y_hat = self(train_data.x, train_data.edge_index, train_data.batch)
        loss = self.criterion(y_hat, train_data.y)
        acc = self.metric(y_hat, train_data.y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_data, batch_idx) -> None:
        y_hat = self(val_data.x, val_data.edge_index, val_data.batch)
        loss = self.criterion(y_hat, val_data.y)
        self.log("val_acc", self.metric(y_hat, val_data.y))
        self.log("val_loss", loss)

    def test_step(self, test_data, batch_idx):
        y_hat = self(test_data.x, test_data.edge_index, test_data.batch)
        error = self.criterion(y_hat, test_data.y)
        self.log("test_acc", self.metric(y_hat, test_data.y))
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINModel")
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser
