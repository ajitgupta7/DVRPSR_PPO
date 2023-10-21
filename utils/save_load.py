import torch
import os
from itertools import zip_longest


def export_train_test_stats(args, start_epoch, train_stats, test_stats):
    fpath = os.path.join(args.output_dir, "loss_gap.csv")
    with open(fpath, 'a') as f:
        f.write((' '.join("{: >16}" for _ in range(8)) + '\n').format(
            "#EP", "#LOSS", "#PROB", "#VAL", "#C_Val", "#TEST_MU", "#TEST_STD", "#TEST_GAP"
        ))
        for epoch, (train, test) in enumerate(zip_longest(train_stats, test_stats), start=start_epoch):
            # train = train if train is not None else ['nan'] * 5
            test = test if test is not None else [0.0, 0.0, 0.0]
            f.write(("{: >16d}" + ' '.join("{: >16.3g}" for _ in range(7)) + '\n').format(
                epoch, *train, *test))


def save_checkpoint(args, epoch, model, optim):
    checkout = {'epoch': epoch,
                'model': model.state_dict(),
                'optim': optim.state_dict()}
    torch.save(checkout, os.path.join(args.output_dir, "epoch{}.pth".format(epoch+1)))


def load_checkpoint(args, model, optim):
    checkout = torch.load(args.resume_state)
    model.load_state_dict(checkout['model'])
    optim.load_state_dict(checkout['optim'])

    return checkout['epoch']

