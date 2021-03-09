import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict
from utils import rot
from utils import get_data, get_teacher_labels, hinge_regression, hinge_classification, get_pca, resize_data
from model import FullyConnected
from config import add_arguments
import argparse

def train_and_test(model, tr_loader, te_loader, crit, task, opt, epochs, checkpoints, device):
    
    tr_losses = []
    tr_accs = []
    te_losses = []
    te_accs = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x,y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            epoch_loss += loss.item()/len(tr_loader)
            opt.step()
        if epoch in checkpoints:
            # tr_losses.append(epoch_loss)
            tr_loss, tr_acc = test(model, tr_loader, crit, task, device)
            te_loss, te_acc = test(model, te_loader, crit, task, device)
            tr_losses.append(tr_loss)
            tr_accs.append(tr_acc)
            te_losses.append(te_loss)
            te_accs.append(te_acc)
            print("Epoch {0} : Train loss {1:.6f}, Train acc {2:.6f}, Test loss {3:.6f}, Test acc {4:.6f}".format(epoch, tr_loss, tr_acc, te_loss, te_acc))
    return tr_losses, tr_accs, te_losses, te_accs

def test(model, te_loader, crit, task, device):    
    with torch.no_grad():
        for (x,y) in te_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_loss = crit(out, y).item()
            if task.endswith('regression'):
                test_acc = 0
            else:
                preds = out.max(1)[1]
                test_acc = preds.eq(y).sum().float()/len(y)
                test_acc = test_acc.item()
    return test_loss, test_acc

def test_ensemble(models, te_loader, crit, task, device):
    with torch.no_grad():
        for (x,y) in te_loader:
            x, y = x.to(device), y.to(device)
            outs = torch.stack([model(x) for model in models])
            out = outs.mean(dim=0)
            test_loss = crit(out, y).item()
            if task.endswith('regression'):
                test_acc = 0
            else:
                preds = out.max(1)[1]
                test_acc = preds.eq(y).sum().float()/len(y)
                test_acc = test_acc.item()
    return test_loss, test_acc

def main(args):

    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.no_cuda: device='cpu'

    if args.task == 'original':
        crit = nn.CrossEntropyLoss()
    else:
        if args.task.endswith('regression'):
            crit = nn.MSELoss()
        elif args.task.endswith('classification'):
            if args.loss_type == 'linear_hinge':
                crit = lambda x,y : hinge_classification(x,y, type='linear')
            elif args.loss_type == 'quadratic_hinge':
                crit = lambda x,y : hinge_classification(x,y, type='quadratic')
            elif args.loss_type == 'nll':
                crit = nn.CrossEntropyLoss()
            else:
                raise NotImplementedError

    torch.manual_seed(0)
    np.random.seed(0)

    if args.dataset == 'random':
        input_size, output_size = args.d, args.num_classes
        tr_data = torch.randn(args.n,args.d)
        te_data = torch.randn(10000, args.d)
        # anisotropy
        d1 = int(args.r_phi * args.d)
        tr_data[:,:args.d - d1] *= args.r_c
        te_data[:,:args.d - d1] *= args.r_c

    else:
        tr_data, te_data, input_size, output_size = get_data(args.dataset, num_classes=args.num_classes, d=args.d, n=args.n, label_noise=args.noise, device=device)
        if args.d is not None:
            input_size = args.d
            if args.pca:
                tr_data, te_data = get_pca(tr_data, te_data, args.d, normalized=args.pca_normalized)
            else:
                tr_data, te_data = resize_data(tr_data, te_data, args.d)
            # anisotropy
            d1 = int(args.r_phi * args.d)
            tr_data.data[:,:args.d - d1] *= args.r_c
            te_data.data[:,:args.d - d1] *= args.r_c

                
    if args.task=='original':
        tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size, shuffle=True)
        te_loader  = torch.utils.data.DataLoader(te_data, batch_size=args.batch_size_test, shuffle=True)
    else:
        if args.dataset != 'random':
            tr_data, te_data = tr_data.data, te_data.data
        teacher = FullyConnected(width=args.teacher_width, n_layers=args.teacher_depth, in_dim=args.d, out_dim=args.num_classes, activation=args.activation).to(device)
        d1 = int(args.r_phi * args.d)
        teacher.layers[0].weight.data[:,:args.d - d1] *= args.r_beta
        with torch.no_grad():
            tr_loader = get_teacher_labels(tr_data, args.task, args.batch_size, args.noise, num_classes=args.num_classes, teacher=teacher)
            te_loader = get_teacher_labels(te_data, args.task, args.batch_size_test, args.noise, num_classes=args.num_classes, teacher=teacher)
        

    tr_losses = []
    tr_accs   = []
    te_losses  = []
    te_accs    = []

    students = []
    checkpoints = np.unique(np.logspace(0,np.log10(args.epochs),20).astype(int))
    for seed in range(args.num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        student = FullyConnected(width=args.width, n_layers=args.depth, in_dim=input_size, out_dim=output_size, activation=args.activation).to(device)

        if args.freeze:
            for p in student.layers[0].parameters():
                p.requires_grad=False
        trainable_parameters = filter(lambda p: p.requires_grad, student.parameters())
        opt = torch.optim.SGD(trainable_parameters, lr=args.lr, momentum=args.mom, weight_decay=args.wd)
        tr_loss_hist, tr_acc_hist, te_loss_hist, te_acc_hist = train_and_test(student, tr_loader, te_loader, crit, args.task, opt, args.epochs, checkpoints, device)
        tr_losses.append(tr_loss_hist)
        tr_accs.append(tr_acc_hist)
        te_losses.append(te_loss_hist)
        te_accs.append(te_acc_hist)
        students.append(student)

    tr_losses, tr_accs, te_losses, te_accs = np.array(tr_losses), np.array(tr_accs), np.array(te_losses), np.array(te_accs)
    tr_loss, tr_acc, te_loss, te_acc = np.mean(tr_losses, axis=0), np.mean(tr_accs, axis=0), np.mean(te_losses, axis=0), np.mean(te_accs, axis=0)
    te_loss_ens, te_acc_ens = test_ensemble(students, te_loader, crit, args.task, device)   
    
    dic = {'args':args, 'checkpoints':checkpoints,
           'tr_loss':tr_loss, 'tr_acc':tr_acc, 'te_loss':te_loss, 'te_acc':te_acc,
           'te_loss_ens':te_loss_ens, 'te_acc_ens':te_acc_ens}
    print(dic)
    torch.save(dic, args.name+'.pyT')
    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)

