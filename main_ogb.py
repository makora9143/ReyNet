import argparse

from pytorch_lightning import Trainer, LightningDataModule, seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader as gDataLoader

from ogb.graphproppred import PygGraphPropPredDataset

from reynet.models.ogb import ReyNetModel, MaronModel, GINModel, Maron2Model, CHOSEN_EPOCH, MARON2_CHOSEN_EPOCH, EGINModel
from reynet.utils import args_print


NUM_FEATURES = 12
NUM_CLASSES = {
    'ogbg-molhiv': 2,
    'ogbg-molpcba': 128,
}


class GraphDataModule(LightningDataModule):
    def __init__(self, dataset: str, batch_size: int):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        PygGraphPropPredDataset(name=self.hparams.dataset, root='./data/')

    def setup(self, stage: str):

        dataset = PygGraphPropPredDataset(name=self.hparams.dataset, root='./data/')

        split_idx = dataset.get_idx_split()

        self.trainset = dataset[split_idx['train']]
        self.valset = dataset[split_idx['valid']]
        self.testset = dataset[split_idx['test']]

        print("train:", self.trainset)
        print("test:", self.testset)

    def train_dataloader(self):
        trainloader = dl(self.trainset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=5)
        return trainloader

    def val_dataloader(self):
        testloader = dl(self.valset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=5)
        return testloader

    def test_dataloader(self):
        testloader = dl(self.testset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=5)
        return testloader


def main():
    seed_everything(1234)

    # Loading Dataset
    dm = Dataset(args.dataset, args.batch_size)

    model = Model(NUM_FEATURES, NUM_CLASSES[args.dataset], args)
    print(model)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    trainer.test(datamodule=dm, ckpt_path="best")
    print(model.result())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='reynet')
    temp_args, _ = parser.parse_known_args()

    if temp_args.model == 'reynet':
        Model = ReyNetModel
    elif temp_args.model == 'maron':
        Model = MaronModel
    elif temp_args.model == 'maron2':
        Model = Maron2Model
    elif temp_args.model == 'gin':
        # Model = GINModel
        Model = EGINModel
    Dataset = GraphDataModule
    dl = gDataLoader

    parser.add_argument('--dataset', '-D', type=str, default='ogbg-molhiv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=str, default=None)

    parser = Model.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    # if args.model == 'gin':
    #     args.max_epochs = 50
    # elif args.model == 'maron2':
    #     args.max_epochs = MARON2_CHOSEN_EPOCH[args.dataset]
    # else:
    #     args.max_epochs = CHOSEN_EPOCH[args.dataset]
    args_print(args)

    main()
