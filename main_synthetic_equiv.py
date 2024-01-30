import argparse

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningDataModule

from reynet.datasets.synthetic import SyntheticDataset, EquivariantDataset
from reynet.models.synthetic_equiv import ReyNetModel, MaronModel, MLPModel
from reynet.utils import args_print


class EquivDataModule(LightningDataModule):
    def __init__(self,
                 dataset: str,
                 set_size: int,
                 num_data: int,
                 batch_size: int = 32,
                 seed: int = 1234):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        trainset: SyntheticDataset = EquivariantDataset(
            self.hparams.dataset,
            self.hparams.set_size,
            train=True,
            seed=self.hparams.seed,
            size=self.hparams.num_data)
        trainloader = DataLoader(trainset,
                                 batch_size=self.hparams.batch_size,
                                 shuffle=True,
                                 num_workers=5)
        return trainloader

    def val_dataloader(self):
        testset: SyntheticDataset = EquivariantDataset(
            self.hparams.dataset,
            self.hparams.set_size,
            train=False,
            seed=self.hparams.seed + 10)

        testloader = DataLoader(testset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers=5)
        return testloader

    def test_dataloader(self):
        testset: SyntheticDataset = EquivariantDataset(
            self.hparams.dataset,
            self.hparams.set_size,
            train=False,
            seed=self.hparams.seed + 10)

        testloader = DataLoader(testset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers=5)
        return testloader


def main(args):

    dm = EquivDataModule(args.dataset, args.set_size, args.num_data,
                         args.batch_size, args.seed)

    model = Model(args)
    print(model)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='reynet')

    temp_args, _ = parser.parse_known_args()

    if temp_args.model == 'reynet':
        Model = ReyNetModel
    elif temp_args.model == 'maron':
        Model = MaronModel
    elif temp_args.model == 'mlp':
        Model = MLPModel

    parser.add_argument('--dataset', '-D', type=str, default='symmetry')
    parser.add_argument('--num-data', '-N', type=int, default=10000)
    parser.add_argument('--set-size', '-S', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch-size', type=int, default=100)

    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args_print(args)
    main(args)
