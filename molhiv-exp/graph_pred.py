import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from utils import Logger, EarlyStopping
from model import EGNN
from reynet import ReyNet

from tqdm.auto import tqdm


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def train(model, device, loader, optimizer, loss_func):
    model.train()

    total_loss = 0
    total = 0

    for batch in tqdm(loader, desc='Train'):
        batch = batch.to(device)

        yh = model(batch)
        optimizer.zero_grad()

        y = batch.y

        # loss = F.mse_loss(yh.float(), y.float())
        # loss = F.binary_cross_entropy_with_logits(yh.float(), y.float())

        if yh.shape[1] > 1:
            loss = loss_func(yh.float(), y.flatten())
        else:
            loss = loss_func(yh.float(), y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += 1

    return total_loss / total


@torch.no_grad()
def eval(model, device, loader, evaluator, eval_metric):
    model.eval()
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc='Eval '):
        batch = batch.to(device)
        y = batch.y

        yh = model(batch)

        if yh.shape[1] > 1:
            y_true.append(y.detach().cpu())
            y_pred.append(torch.argmax(yh.detach(), dim=1).view(-1, 1).cpu())
        else:
            y_true.append(y.view(yh.shape).detach().cpu())
            y_pred.append(yh.detach().cpu())

        # y_true.append(y.view(yh.shape).detach().cpu())
        # y_pred.append(yh.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {'y_true': y_true, 'y_pred': y_pred}

    return evaluator.eval(input_dict)[eval_metric]


def run_graph_pred(args, model, dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model.to(device)
    model.set_device(device)

    evaluator = Evaluator(name=args.dataset)

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    logger = Logger(args.runs, mode='max')

    for run in range(args.runs):
        tqdm.write('\nRun {}'.format(run + 1))
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.dataset == 'ogbg-molhiv':
            loss_func = F.binary_cross_entropy_with_logits
        elif args.dataset == 'ogbg-ppa':
            loss_func = F.cross_entropy

        early_stopping = EarlyStopping(
            patience=args.patience, verbose=True, mode='max')

        for epoch in range(1, 1 + args.epochs):
            tqdm.write('epoch {}'.format(epoch))
            loss = train(model, device, train_loader, optimizer, loss_func)

            train_metric = eval(model, device, train_loader,
                                evaluator, dataset.eval_metric)
            valid_metric = eval(model, device, valid_loader,
                                evaluator, dataset.eval_metric)
            test_metric = eval(model, device, test_loader,
                               evaluator, dataset.eval_metric)

            result = [train_metric, valid_metric, test_metric]

            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                tqdm.write(
                    f'Loss: {loss:.4f}, '
                    f'Train: {train_metric:.4f}, '
                    f'Valid: {valid_metric:.4f} '
                    f'Test: {test_metric:.4f}')
                tqdm.write("\n")

            if early_stopping(valid_metric, model):
                break

        logger.print_statistics(run)
    logger.print_statistics()


def main():
    parser = argparse.ArgumentParser(
        description='train graph property prediction')
    parser.add_argument('--dataset', type=str, default='ogbg-molhiv',
                        choices=['ogbg-molhiv', 'ogbg-ppa'])
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='path to dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--model',
                        type=str,
                        default='gcn',
                        choices=["gcn", "gin", "reynet"])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--setbn', type=int, default=0)
    parser.add_argument('--chbn', type=int, default=0)
    parser.add_argument('--setchbn', type=int, default=0)
    parser.add_argument('--lastbn', type=int, default=0)
    args = parser.parse_args()
    print(args)
    print(torch.cuda.device_count())

    if args.dataset == 'ogbg-molhiv':
        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          root=args.dataset_path)

        if args.model in ["gcn", "gin"]:
            model = EGNN(args.hidden_channels,
                         dataset.num_tasks,
                         args.num_layers,
                         args.dropout,
                         args.model,
                         mol=True)
        elif args.model == "reynet":
            model = ReyNet(
                args.hidden_channels, dataset.num_tasks, args.num_layers, dropout=args.dropout, setbn=args.setbn, chbn=args.chbn, setchbn=args.setchbn, lastbn=args.lastbn, skip=True, mol=True)
    elif args.dataset == 'ogbg-ppa':
        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          root=args.dataset_path,
                                          transform=add_zeros)

        if args.model in ["gcn", "gin"]:
            model = EGNN(args.hidden_channels, int(dataset.num_classes),
                         args.num_layers, args.dropout, args.model)
        elif args.model == "reynet":
            model = None

    print(model)
    run_graph_pred(args, model, dataset)


if __name__ == '__main__':
    main()
