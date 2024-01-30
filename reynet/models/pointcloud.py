from torch import Tensor
from torch import nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics import Accuracy

from ..layers.equiv.reynolds import ReyEquiv1to1
from ..layers.equiv.zaheer import PermEqui1_max
from ..layers.pooling import Sumpool, Maxpool


class Subtract(nn.Module):
    def __init__(self, in_features, set_layers, channel_layers, activation=nn.Tanh()):
        super().__init__()
        self.pooling = Maxpool(-1, keepdim=True)
        self.operator = ReyEquiv1to1(in_features, set_layers=set_layers, channel_layers=channel_layers, activation=nn.Tanh())

    def forward(self, x):
        xm = self.pooling(x)
        return self.operator(x - xm)


class ReyNetModel(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.model = nn.Sequential(
            ReyEquiv1to1(3,
                         set_layers=[8, 8],
                         channel_layers=[64, 128, 256],
                         activation=nn.ReLU(inplace=True),
                         skip_connection=self.args.use_skip),
            Sumpool(dim=-1),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, int(args.name))
        )

        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr,
                               weight_decay=1e-5)

        schedular = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(
                range(self.args.decay, self.args.max_epochs, self.args.decay)),
            gamma=0.1)
        return [optimizer], [schedular]

    def training_step(self, train_batch, batch_idx):
        x = train_batch.pos.reshape(train_batch.num_graphs, 100, 3).transpose(-1, -2)
        y = train_batch.y
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)
        self.log("train_loss", loss, batch_size=train_batch.num_graphs)
        self.log("train_acc", acc, on_epoch=True, batch_size=train_batch.num_graphs)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch.pos.reshape(val_batch.num_graphs, 100, 3).transpose(-1, -2)
        y = val_batch.y
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x = test_batch.pos.reshape(test_batch.num_graphs, 100, 3).transpose(-1, -2)
        y = test_batch.y
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("test_acc", self.metric(y_hat, y))
        self.log("test_loss", loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ReyNetModel")
        parser.add_argument('--skip', action='store_true')
        parser.add_argument('--decay', type=int, default=400)
        parser.add_argument('--use-skip', action='store_true')

        parser.add_argument('--lr', type=float, default=1e-3)
        return parent_parser


class DeepsetsModel(pl.LightningModule):
    def __init__(self, channel_layers, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        if args.use_max:
            module = PermEqui1_max
        else:
            module = nn.Linear
        self.phi = nn.Sequential(
            module(3, 256),
            nn.Tanh(),
            module(256, 256),
            nn.Tanh(),
            module(256, 256),
            nn.Tanh(),
            Maxpool(1),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(256, int(args.name))
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        return self.phi(x)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(),
                               lr=self.args.lr,
                               weight_decay=1e-5)

        schedular = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(
                range(self.args.decay, self.args.max_epochs, self.args.decay)),
            gamma=0.1)
        return [optimizer], [schedular]

    def training_step(self, train_batch, batch_idx):
        x = train_batch.pos.reshape(train_batch.num_graphs, 100, 3)
        y = train_batch.y
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.metric(y_hat, y)
        self.log("train_loss", loss, batch_size=train_batch.num_graphs)
        self.log("train_acc", acc, on_epoch=True, batch_size=train_batch.num_graphs)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch.pos.reshape(val_batch.num_graphs, 100, 3)
        y = val_batch.y
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.metric(y_hat, y))

    def test_step(self, test_batch, batch_idx):
        x = test_batch.pos.reshape(test_batch.num_graphs, 100, 3).transpose(-1, -2)
        y = test_batch.y
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_acc", self.metric(y_hat, y))
        self.log("test_loss", loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DeepsetModel")
        parser.add_argument('--channel-layers', type=int, nargs='+', default=[64])
        parser.add_argument('--decay', type=int, default=400)
        parser.add_argument('--use-max', action='store_true')

        parser.add_argument('--lr', type=float, default=1e-3)
        return parent_parser
