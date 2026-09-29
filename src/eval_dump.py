# -*- coding: utf-8 -*-
"""Re-evaluate trained EpiGNN checkpoints and dump per-timestep test predictions.

EpiGNN already scores a single lead-h target, so unlike the colagnn baselines its
protocol needs no correction. What is unverified is the train/val/test split the
checkpoints were produced under: this repo's CLI defaults are .5/.2/.3 while the
manuscript reports .6/.2/.2. Each checkpoint is therefore evaluated under both
splits so the published numbers can be attributed to one of them.

Run from the repository root:  python src/eval_dump.py
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import csv
import os
import re
import sys
from math import sqrt

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                            mean_squared_error, r2_score)

ap = argparse.ArgumentParser()
ap.add_argument('--save_dir', type=str, default='save')
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--out_csv', type=str, default='result/split_comparison.csv')
ap.add_argument('--pred_dir', type=str, default='../MSAGAT-Net/report/predictions')
ap.add_argument('--dump_split', type=str, default='0.6',
                help='which split to dump predictions for: 0.6 or 0.5')
cli = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.gpu)

import torch  # noqa: E402

from data import DataBasicLoader  # noqa: E402
from models import EpiGNN  # noqa: E402

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SIM_MAT = {
    'japan': 'japan-adj',
    'region785': 'region-adj',
    'state360': 'state-adj-49',
    'australia-covid': 'australia-adj',
    'ltla_timeseries': 'ltla-adj',
    'nhs_timeseries': 'nhs-adj',
}

SPLITS = {'0.6': (.6, .2, .2), '0.5': (.5, .2, .3)}

CKPT_RE = re.compile(r'^EpiGNN\.(?P<dataset>.+)\.w-(?P<window>\d+)'
                     r'\.h-(?P<horizon>\d+)\.pt$')


class Args(object):
    """Mirror of train.py's argparse defaults, restricted to what is read here."""

    def __init__(self, dataset, horizon, window, split):
        self.dataset = dataset
        self.sim_mat = SIM_MAT[dataset]
        self.horizon = horizon
        self.window = window
        self.train, self.val, self.test = SPLITS[split]
        self.n_layer, self.n_hidden = 1, 20
        self.dropout = 0.2
        self.batch = 128
        self.seed = 42
        self.cuda = torch.cuda.is_available()
        self.gpu = 0
        self.save_dir = 'save'
        self.k, self.hidR, self.hidA, self.hidP = 8, 64, 64, 1
        self.hw = 0
        self.n, self.res, self.s = 2, 0, 2
        self.extra, self.label, self.pcc = '', '', ''
        self.ablation = None
        self.model = 'EpiGNN'
        self.lamda = 0.01
        self.shuffle = False
        self.mylog = False


def metrics(y_true_states, y_pred_states):
    """Reproduce EpiGNN train.py:179-214."""
    rmse_states = np.mean(np.sqrt(mean_squared_error(
        y_true_states, y_pred_states, multioutput='raw_values')))
    pcc_states = np.mean(np.array(
        [pearsonr(y_true_states[:, k], y_pred_states[:, k])[0]
         for k in range(y_true_states.shape[1])]))
    r2_states = np.mean(r2_score(y_true_states, y_pred_states,
                                 multioutput='raw_values'))
    flat_true = np.reshape(y_true_states, (-1))
    flat_pred = np.reshape(y_pred_states, (-1))
    return {
        'mae': mean_absolute_error(flat_true, flat_pred),
        'rmse': sqrt(mean_squared_error(flat_true, flat_pred)),
        'rmse_states': rmse_states,
        'pcc': pearsonr(flat_true, flat_pred)[0],
        'pcc_states': pcc_states,
        'r2': r2_score(flat_true, flat_pred, multioutput='uniform_average'),
        'r2_states': r2_states,
        'var': explained_variance_score(flat_true, flat_pred,
                                        multioutput='uniform_average'),
    }


def evaluate_checkpoint(dataset, horizon, window, split, ckpt_path):
    args = Args(dataset, horizon, window, split)
    loader = DataBasicLoader(args, DEVICE)

    model = EpiGNN(args, loader).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader.get_batches(loader.test, args.batch, False):
            X, Y, index = batch[0], batch[1], batch[2]
            out, _ = model(X, index)
            preds.append(out.cpu())
            trues.append(Y.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()
    scale, shift = (loader.max - loader.min), loader.min
    y_true_s = y_true * scale + shift
    y_pred_s = y_pred * scale + shift
    return metrics(y_true_s, y_pred_s), y_true_s, y_pred_s


def main():
    ckpts = sorted(f for f in os.listdir(cli.save_dir) if f.endswith('.pt'))
    rows = []

    for fname in ckpts:
        m = CKPT_RE.match(fname)
        if m is None:
            print('SKIP (unparsed name): %s' % fname)
            continue
        dataset = m.group('dataset')
        horizon = int(m.group('horizon'))
        window = int(m.group('window'))
        if dataset not in SIM_MAT:
            continue

        path = os.path.join(cli.save_dir, fname)
        row = {'dataset': dataset, 'horizon': horizon}
        for split in ('0.6', '0.5'):
            res, yt, yp = evaluate_checkpoint(dataset, horizon, window,
                                              split, path)
            row['rmse_%s' % split] = res['rmse']
            row['mae_%s' % split] = res['mae']
            row['pcc_%s' % split] = res['pcc']
            row['r2_%s' % split] = res['r2']
            row['n_test_%s' % split] = yt.shape[0]

            if split == cli.dump_split:
                outdir = os.path.join(cli.pred_dir, dataset)
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
                token = 'epignn.%s.w-%d.h-%d.none.seed-42.oldckpt' % (
                    dataset, window, horizon)
                np.savez_compressed(
                    os.path.join(outdir, token + '.npz'),
                    y_true=yt, y_pred=yp, model='epignn', dataset=dataset,
                    horizon=horizon, window=window, seed=42, ablation='none',
                    protocol='lead_h', split=split)

        rows.append(row)
        print('epignn %-18s h=%-3d  RMSE @.6/.2/.2 = %11.4f   '
              '@.5/.2/.3 = %11.4f' % (dataset, horizon,
                                      row['rmse_0.6'], row['rmse_0.5']))

    if not rows:
        print('No checkpoints evaluated.')
        return 1

    outdir = os.path.dirname(cli.out_csv)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(cli.out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\nWrote %d rows to %s' % (len(rows), cli.out_csv))
    return 0


if __name__ == '__main__':
    sys.exit(main())
