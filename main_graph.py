import argparse

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningDataModule, seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader as gDataLoader

from reynet.datasets.graph import GraphDataset, NUM_LABELS, NUM_CLASSES
from reynet.datasets.graph import get_tudataset
from reynet.models.graph import ReyNetModel, MaronModel, GINModel, Maron2Model, CHOSEN_EPOCH, MARON2_CHOSEN_EPOCH
from reynet.utils import args_print


class GraphDataModule(LightningDataModule):
    def __init__(self, dataset: str, batch_size: int, num_fold: int = 0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.trainset = GraphDataset(self.hparams.dataset,
                                     train=True,
                                     batch_size=self.hparams.batch_size,
                                     num_fold=self.hparams.num_fold)
        self.valset = GraphDataset(self.hparams.dataset,
                                   train=False,
                                   batch_size=self.hparams.batch_size,
                                   num_fold=self.hparams.num_fold)
        self.testset = GraphDataset(self.hparams.dataset,
                                    train=False,
                                    batch_size=self.hparams.batch_size,
                                    num_fold=self.hparams.num_fold)
        print(self.trainset)
        print(self.testset)

    def train_dataloader(self):
        trainloader = dl(self.trainset, batch_size=None, num_workers=5)
        return trainloader

    def val_dataloader(self):
        testloader = dl(self.valset, batch_size=None, num_workers=5)
        return testloader

    def test_dataloader(self):
        testloader = dl(self.testset, batch_size=None, num_workers=5)
        return testloader


class GraphMessageDataModule(LightningDataModule):
    def __init__(self, dataset: str, batch_size: int, num_fold: int = 0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        if self.hparams.dataset == "IMDBBINARY":
            degree = 135
        elif self.hparams.dataset == "IMDBMULTI":
            degree = 88
        elif self.hparams.dataset == "COLLAB":
            degree = 491

        self.trainset, self.testset = get_tudataset(
            self.hparams.dataset,
            num_fold=self.hparams.num_fold,
            pre_transform=T.OneHotDegree(degree)
            if self.hparams.dataset in ["IMDBBINARY", "IMDBMULTI", "COLLAB"] else None)
        print(self.trainset)
        print(self.testset)

    def train_dataloader(self):
        trainloader = dl(self.trainset, batch_size=self.hparams.batch_size, num_workers=5)
        return trainloader

    def val_dataloader(self):
        testloader = dl(self.testset,
                        batch_size=self.hparams.batch_size,
                        num_workers=5)
        return testloader

    def test_dataloader(self):
        testloader = dl(self.testset,
                        batch_size=self.hparams.batch_size,
                        num_workers=5)
        return testloader


def main():
    seed_everything(1234)

    # Loading Dataset
    dm = Dataset(args.dataset, args.batch_size, args.num_fold)

    if args.model == 'gin':
        if args.dataset == "IMDBBINARY":
            degree = 135
        elif args.dataset == "IMDBMULTI":
            degree = 88
        elif args.dataset == "COLLAB":
            degree = 491
        in_features = NUM_LABELS[args.dataset] + 2 + degree
    else:
        in_features = NUM_LABELS[args.dataset] + 1

    model = Model(in_features, NUM_CLASSES[args.dataset], args)
    print(model)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='reynet')
    temp_args, _ = parser.parse_known_args()

    if temp_args.model == 'reynet':
        Model = ReyNetModel
        Dataset = GraphDataModule
        dl = DataLoader
    elif temp_args.model == 'maron':
        Model = MaronModel
        Dataset = GraphDataModule
        dl = DataLoader
    elif temp_args.model == 'maron2':
        Model = Maron2Model
        Dataset = GraphDataModule
        dl = DataLoader
    elif temp_args.model == 'gin':
        Model = GINModel
        Dataset = GraphMessageDataModule
        dl = gDataLoader

    parser.add_argument('--dataset', '-D', type=str, default='MUTAG')
    parser.add_argument('--num-fold', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    if args.model == 'gin':
        args.max_epochs = 50
    elif args.model == 'maron2':
        args.max_epochs = MARON2_CHOSEN_EPOCH[args.dataset]
    else:
        args.max_epochs = CHOSEN_EPOCH[args.dataset]
    args_print(args)

    main()
