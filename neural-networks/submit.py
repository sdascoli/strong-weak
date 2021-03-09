import torch
import numpy as np
import submitit
from pathlib import Path
from main import main
import collections
import itertools
import time
from argparse import Namespace
import os

#get default arguments
from config import add_arguments
import argparse
from argparse import Namespace
def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults
parser = argparse.ArgumentParser()
parser = add_arguments(parser)
args = get_argparse_defaults(parser)


folder = '/checkpoint/sdascoli/nn-anisotropy/r.{}/'.format(int(time.time()))
if not os.path.exists(folder):
    os.mkdir(folder)

widths = np.unique(np.logspace(1, 9, 15, base=2).astype(int))
ns = np.logspace(7, 15, 15, base=2).astype(int)
grid = collections.OrderedDict({
    'width' : widths,
    'n': ns,
    'depth': [1],
    'wd' : [0],
    'activation' : ['relu'],
    'dataset' : ['MNIST','CIFAR10'],
    'pca' : [0,1],
    'lr' : [0.05],
    'mom' : [0.9],
    'batch_size' : [1000000],
    'd' : [100],
    'num_seeds': [10],
    'noise' : [0, 0.5],
    'test_noise' : [False],
    'epochs' : [3000],
    'no_cuda' : [True],
    })

def dict_product(d):
    keys = d.keys()
    for element in itertools.product(*d.values()):
        yield dict(zip(keys, element))

torch.save(grid, folder + '/params.pkl')

ex = submitit.AutoExecutor(folder)
ex.update_parameters(name="anisotropy")
ex.update_parameters(cpus_per_task=2, timeout_min=3000)
jobs = []
with ex.batch():
    for i, params in enumerate(dict_product(grid)):
        params['name'] = folder+'/{:06d}'.format(i)
        for k,v in params.items():
            args[k] = v
        job = ex.submit(main, Namespace(**args))
        jobs.append(job)

