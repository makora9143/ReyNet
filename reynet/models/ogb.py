from torch import Tensor
from torch import nn
from torch import optim
import torch

import pytorch_lightning as pl

from torchmetrics import Accuracy, AUROC

from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import GIN, global_add_pool

from ..layers.equiv.reynolds import ReyEquiv2to2
from ..layers.equiv.maron import MaronEquivNet2to2, RegularBlock
from ..layers.pooling import DiagOffdiagSumpool

from .ogb_baseline import EGNN
from ogb.graphproppred import Evaluator


MARON_LEARNING_RATES = {
    'COLLAB': 0.00256,
    'IMDBBINARY': 0.00064,
    'IMDBMULTI': 0.00064,
    'MUTAG': 0.00512,
    'NCI1': 1e-4,
    'NCI109': 1e-4,
    'PROTEINS': 0.00064,
    'PTC': 0.00128
}
REYNET_LEARNING_RATES = {
    'COLLAB': 1e-4,
    'IMDBBINARY': 1e-4,
    'IMDBMULTI': 1e-4,
    'MUTAG': 1e-3,
    'NCI1': 1e-4,
    'NCI109': 1e-4,
    'PROTEINS': 1e-4,
    'PTC': 1e-4,
    'ogbg-molhiv': 1e-3,
}
DECAY_RATES = {
    'COLLAB': 0.7,
    'IMDBBINARY': 0.4,
    'IMDBMULTI': 0.7,
    'MUTAG': 0.7,
    'NCI1': 0.7,
    'NCI109': 0.7,
    'PROTEINS': 0.7,
    'PTC': 0.6,
    'ogbg-molhiv': 0.7,
}
CHOSEN_EPOCH = {
    'COLLAB': 100,
    'IMDBBINARY': 40,
    'IMDBMULTI': 150,
    'MUTAG': 130,
    'NCI1': 99,
    'NCI109': 99,
    'PROTEINS': 20,
    'PTC': 9,
    'ogbg-molhiv': 130,
}

MARON2_LEARNING_RATES = {
    'COLLAB': 0.0001,
    'IMDBBINARY': 0.00001,
    'IMDBMULTI': 0.0001,
    'MUTAG': 0.0005,
    'NCI1': 0.0005,
    'NCI109': 0.0001,
    'PROTEINS': 0.0005,
    'PTC': 0.001
}
MARON2_DECAY_RATES = {
    'COLLAB': 0.5,
    'IMDBBINARY': 0.75,
    'IMDBMULTI': 1.0,
    'MUTAG': 0.5,
    'NCI1': 0.75,
    'NCI109': 1.0,
    'PROTEINS': 0.75,
    'PTC': 0.5
}
MARON2_CHOSEN_EPOCH = {
    'COLLAB': 85,
    'IMDBBINARY': 100,
    'IMDBMULTI': 150,
    'MUTAG': 150,
    'NCI1': 100,
    'NCI109': 300,
    'PROTEINS': 100,
    'PTC': 200
}


def format_batch(batch, use_edge_attr: bool = False):
    batch_idx = batch.batch
    return torch.cat([
        torch.diag_embed(to_dense_batch(batch.x, batch_idx)[0].transpose(-1, -2)),
        to_dense_adj(batch.edge_index, batch_idx, batch.edge_attr if use_edge_attr else None).permute(0, 3, 1, 2)], 1), batch.y.squeeze()


class ReyNetModel(pl.LightningModule):
    def __init__(self, in_features, num_classes, args) -> None:
        super().__init__()
        self.args = args

        self.model = nn.Sequential(
            ReyEquiv2to2(in_features,
                         set_layers=[16, 16, 16],
                         channel_layers=[16, 32, 256],
                         skip_connection=self.args.residual),
            DiagOffdiagSumpool(),
            nn.Linear(2 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes))

        def initialize_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
        self.model.apply(initialize_weights)

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()
        self.evaluator = Evaluator(name='ogbg-molhiv')
        self.test_y_true = []
        self.test_y_pred = []

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=REYNET_LEARNING_RATES[self.args.dataset])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATES[self.args.dataset])
        print(optimizer)
        print(scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 20,
            }
        }

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = format_batch(train_batch, True)
        x = x.float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = format_batch(val_batch, True)
        x = x.float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_acc", self.metric(y_hat, y))
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = format_batch(test_batch, True)
        x = x.float()
        y_hat = self.model(x)
        error = self.criterion(y_hat, y)
        self.log("test_acc", self.metric(y_hat, y))
        self.log("test_error", error)
        self.test_y_true.append(test_batch.y.detach().cpu())
        self.test_y_pred.append(torch.argmax(y_hat.detach(), dim=1).view(-1, 1).cpu())

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ReyNetModel")
        parser.add_argument('--set-layers', type=int, nargs='+', default=[128])
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        parser.add_argument('--gamma', type=float, default=1e-3)
        parser.add_argument('--residual', action='store_true')
        return parent_parser

    def result(self):
        y_true = torch.cat(self.test_y_true, dim=0).numpy()
        y_pred = torch.cat(self.test_y_pred, dim=0).numpy()
        print(y_true.shape, y_pred.shape)

        input_dict = {'y_true': y_true, 'y_pred': y_pred}

        return self.evaluator.eval(input_dict)['rocauc']


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
        optimizer = optim.Adam(self.parameters(), lr=MARON_LEARNING_RATES[self.args.dataset])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATES[self.args.dataset])
        print(optimizer)
        print(scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 20,
            }
        }

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = format_batch(train_batch)
        x = x.float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = format_batch(val_batch)
        x, y = x.float(), y.unsqueeze(0)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("val_acc", self.metric(y_hat, y))
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        x, y = format_batch(test_batch)
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
        parser.add_argument('--gamma', type=float, default=1e-3)
        return parent_parser


class Maron2Model(pl.LightningModule):
    def __init__(self, in_features, num_classes, args) -> None:
        super().__init__()
        self.args = args

        self.model = nn.Sequential(
            RegularBlock(in_features, 256),
            RegularBlock(256, 256),
            RegularBlock(256, 256),
            DiagOffdiagSumpool(),
            nn.Linear(2 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        preds = self.model(x)
        return preds

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=MARON2_LEARNING_RATES[self.args.dataset])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=MARON2_DECAY_RATES[self.args.dataset])
        print(optimizer)
        print(scheduler)
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
        parser.add_argument('--gamma', type=float, default=1e-3)
        return parent_parser


class GINModel(pl.LightningModule):
    def __init__(self, in_features, num_classes, args) -> None:
        super().__init__()
        self.args = args

        self.gnn = GIN(9, 64, 5, norm=nn.BatchNorm1d(64), dropout=0.5, jk='cat')
        self.classifier = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()
        self.auroc = AUROC(num_classes=num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, batch) -> Tensor:
        h = self.gnn(x.float(), edge_index)
        h = global_add_pool(h, batch)
        h = self.classifier(h)
        return h

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def training_step(self, train_data, batch_idx) -> Tensor:
        y_hat = self(train_data.x, train_data.edge_index, train_data.batch)
        y = train_data.y.flatten()
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)
        auroc = self.auroc(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_auroc", auroc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_data, batch_idx) -> None:
        y_hat = self(val_data.x, val_data.edge_index, val_data.batch)
        y = val_data.y.flatten()
        loss = self.criterion(y_hat, y)
        self.log("val_acc", self.metric(y_hat, y))
        self.log("val_auroc", self.auroc(y_hat, y))
        self.log("val_loss", loss)

    def test_step(self, test_data, batch_idx):
        y_hat = self(test_data.x, test_data.edge_index, test_data.batch)
        y = test_data.y.flatten()
        error = self.criterion(y_hat, y)
        self.log("test_acc", self.metric(y_hat, y))
        self.log("test_auroc", self.auroc(y_hat, y))
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINModel")
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser




class EGINModel(pl.LightningModule):
    def __init__(self, in_features, num_classes, args) -> None:
        super().__init__()
        self.args = args

        self.model = EGNN(300, 2, 5, 0.5, 'gcn', mol=True)

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()
        self.evaluator = Evaluator(name='ogbg-molhiv')
        self.test_y_true = []
        self.test_y_pred = []

    def forward(self, batch) -> Tensor:
        return self.model(batch)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def training_step(self, train_data, batch_idx) -> Tensor:
        y_hat = self(train_data)
        y = train_data.y.flatten()
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_data, batch_idx) -> None:
        y_hat = self(val_data)
        y = val_data.y.flatten()
        loss = self.criterion(y_hat, y)
        self.log("val_acc", self.metric(y_hat, y))
        self.log("val_loss", loss)

    def test_step(self, test_data, batch_idx):
        y_hat = self(test_data)
        y = test_data.y.flatten()
        error = self.criterion(y_hat, y)

        self.test_y_true.append(test_data.y.detach().cpu())
        self.test_y_pred.append(torch.argmax(y_hat.detach(), dim=1).view(-1, 1).cpu())

        self.log("test_acc", self.metric(y_hat, y))
        self.log("test_error", error)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINModel")
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])

        parser.add_argument('--learning-rate', type=float, default=1e-3)
        return parent_parser

    def result(self):
        y_true = torch.cat(self.test_y_true, dim=0).numpy()
        y_pred = torch.cat(self.test_y_pred, dim=0).numpy()
        print(y_true.shape, y_pred.shape)

        input_dict = {'y_true': y_true, 'y_pred': y_pred}

        return self.evaluator.eval(input_dict)['rocauc']
