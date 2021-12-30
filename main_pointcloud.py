import argparse

import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from pytorch_lightning import Trainer, LightningDataModule, seed_everything

from reynet.models.pointcloud import ReyNetModel, DeepsetsModel
from reynet.utils import args_print


class ModelNetDataModule(LightningDataModule):
    def __init__(self, name: str, batch_size: int):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        train_transform = T.Compose([
            T.RandomScale((0.8, 1.25)),
            T.RandomRotate(18, axis=2),
            T.SamplePoints(100),
        ])
        self.trainset = ModelNet(f"./data/ModelNet{self.hparams.name}",
                                 name=self.hparams.name,
                                 train=True,
                                 transform=train_transform,
                                 pre_transform=T.NormalizeScale())
        self.testset = ModelNet(f"./data/ModelNet{self.hparams.name}",
                                name=self.hparams.name,
                                train=False,
                                transform=T.SamplePoints(100),
                                pre_transform=T.NormalizeScale())

    def train_dataloader(self):
        trainloader = DataLoader(self.trainset,
                                 batch_size=self.hparams.batch_size,
                                 shuffle=True,
                                 num_workers=8)
        return trainloader

    def val_dataloader(self):
        testloader = DataLoader(self.testset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers=8)
        return testloader

    def test_dataloader(self):
        testloader = DataLoader(self.testset,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers=8)
        return testloader


def main():
    seed_everything(1234)
    # Loading Dataset
    dm = ModelNetDataModule(args.name, args.batch_size)

    if args.model == 'reynet':
        model = Model(args.set_layers, args.channel_layers, args)
    elif args.model == 'deepset':
        model = Model(args.channel_layers, args)
    print(model)
    trainer = Trainer.from_argparse_args(args, gradient_clip_val=5)
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path='best')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='reynet')
    temp_args, _ = parser.parse_known_args()

    if temp_args.model == 'reynet':
        Model = ReyNetModel
    elif temp_args.model == 'deepset':
        Model = DeepsetsModel

    parser.add_argument('--name', type=str, default="40")
    parser.add_argument('--batch-size', type=int, default=64)

    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args_print(args)

    main()
