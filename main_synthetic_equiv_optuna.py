import argparse
from typing import List

import torch
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, LightningDataModule, LightningModule

import optuna
from optuna.integration import PyTorchLightningPruningCallback

from reynet.datasets.synthetic import SyntheticDataset, EquivariantDataset
from reynet.layers.equiv.sannai import ReyEquiv2to2
from reynet.loss import CornerMSELoss


class ReyNetModel(LightningModule):
    def __init__(self, set_layers: List[int], channel_layers: List[int], args) -> None:
        super().__init__()
        self.args = args

        self.model = ReyEquiv2to2(1,
                                  set_layers=set_layers,
                                  channel_layers=channel_layers + [1])
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


class EquivDataModule(LightningDataModule):
    def __init__(self,
                 dataset: str,
                 set_size: int,
                 num_data: int,
                 batch_size: int = 32,
                 seed: int = 1234):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.trainset: SyntheticDataset = EquivariantDataset(
            self.hparams.dataset,
            self.hparams.set_size,
            train=True,
            seed=self.hparams.seed,
            size=self.hparams.num_data)

        self.valset: SyntheticDataset = EquivariantDataset(
            self.hparams.dataset,
            self.hparams.set_size,
            train=False,
            seed=self.hparams.seed + 10)

        self.testset: SyntheticDataset = EquivariantDataset(
            self.hparams.dataset,
            self.hparams.set_size,
            train=False,
            seed=self.hparams.seed + 20)

    def train_dataloader(self):
        trainloader = DataLoader(self.trainset,
                                 batch_size=self.hparams.batch_size,
                                 shuffle=True,
                                 num_workers=5)
        return trainloader

    def val_dataloader(self):
        valloader = DataLoader(self.valset,
                               batch_size=self.hparams.batch_size,
                               shuffle=False,
                               num_workers=5)
        return valloader

    def test_dataloader(self):
        testloader = DataLoader(self.testset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers=5)
        return testloader


def objective(trial: optuna.trial.Trial) -> float:

    dm = EquivDataModule(args.dataset, args.set_size, args.num_data,
                         args.batch_size, args.seed)

    n_set_layers = trial.suggest_int("n_set_layers", 1, 3)
    output_dims_set_layers = [
        trial.suggest_int("n_units_set_l{}".format(i), 4, 256, log=True) for i in range(n_set_layers)
    ]
    n_channel_layers = trial.suggest_int("n_channel_layers", 1, 3)
    output_dims_channel_layers = [
        trial.suggest_int("n_units_channel_l{}".format(i), 4, 256, log=True) for i in range(n_channel_layers)
    ]

    model = Model(output_dims_set_layers, output_dims_channel_layers, args)
    print(model)
    trainer = Trainer.from_argparse_args(args, callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_error')])
    trainer.fit(model, datamodule=dm)
    # trainer.test(datamodule=dm, ckpt_path="best")

    return trainer.callback_metrics['val_error'].item()


if __name__ == '__main__':
    Model = ReyNetModel

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='reynet')
    parser.add_argument('--dataset', '-D', type=str, default='symmetry')
    parser.add_argument('--num-data', '-N', type=int, default=10000)
    parser.add_argument('--set-size', '-S', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch-size', type=int, default=100)

    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )

    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=43200)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
