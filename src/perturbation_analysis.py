import warnings
warnings.filterwarnings('ignore')
import os
import sys
import argparse
import random
import joblib
import json 

import numpy as np
import pandas as pd
import sklearn

from types import SimpleNamespace
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn, optim

import train as this

from geomloss import SamplesLoss
from annoy import AnnoyIndex


def perturb(args, std, pca):                        
    x=std["data"]
    scaler=sklearn.preprocessing.StandardScaler()
    x=scaler.fit_transform(x)
    x_ = x
    genes = std["genes"]
    perturb_genes=args.perturb_genes.split(",")

    idx=[]
    for elt in perturb_genes:
        if (elt in genes):
            idx.append(genes.index(elt))
    for elt in idx:
        x_[:,elt]= args.z_score 
    xp = pca.transform(x_)
    return xp

def simulate(args, device, model, config, x_std, x_perturb, meta):
    x_std=torch.from_numpy(x_std)
    x_perturb=torch.from_numpy(x_perturb)
    num_sims=args.num_sims
    num_init=args.num_init
    num_steps = args.num_steps
    day=args.day
    
    all_sims = []
    all_sims_perturb = []
    for _ in range(num_sims):
        if args.study=="veres":
            if args.cell_type!=None:
                idx=pd.DataFrame(meta[meta["Annotation"]==args.cell_type]).sample(num_init, weights="growth").index
            elif args.day!=None:
                idx=pd.DataFrame(meta[(meta["Time point"]==day)]).sample(num_init,weights="growth").index
        if args.study=="weinreb":
            idx=pd.DataFrame(meta[(meta["Annotation"]=="undiff") & (meta["Time point"]==2)]).sample(num_init, weights="growth").index

        x_i = x_std[idx].to(device) 
        x_i_ = x_i.detach().cpu().numpy()

        x_i_perturb = x_perturb[idx].to(device) #.reshape(1,-1).to(device)
        x_i_perturb_ = x_i_perturb.detach().cpu().numpy()

        xps_i = [x_i_]
        xps_i_perturb = [x_i_perturb_]

        for _ in range(num_steps):
            z = torch.randn(x_i.shape[0], x_i.shape[1]) * config.train_sd
            z = z.to(device)
            x_i = model._step(x_i.float(), dt=config.train_dt, z=z)
            x_i_ = x_i.detach().cpu().numpy()
            xps_i.append(x_i_)

            z = torch.randn(x_i_perturb.shape[0], x_i_perturb.shape[1]) * config.train_sd
            z = z.to(device)
            x_i_perturb = model._step(x_i_perturb.float(), dt=config.train_dt, z=z)
            x_i_perturb_ = x_i_perturb.detach().cpu().numpy()
            xps_i_perturb.append(x_i_perturb_)

        xps = np.stack(xps_i)
        xps_perturb = np.stack(xps_i_perturb)
        all_sims.append(xps)
        all_sims_perturb.append(xps_perturb)
    return all_sims, all_sims_perturb

def classify_cells(args, all_sims_timepoints, ann_dir, std):
    n_neighbors=10
    meta = std["meta"]
    yc = meta["Annotation"]
    xp_df = pd.DataFrame(std["xp_std"], yc) 
    u = AnnoyIndex(all_sims_timepoints[0][0].shape[1], 'euclidean')  # all_sims_timepoints[0][0][0].shape[1], 'euclidean')
    u.load(ann_dir)
    yp_all=[]
    for timepoint in all_sims_timepoints:
        yp=[]
        for i in range(len(timepoint)):
            yt=[]
            for j in range(len(timepoint[0])):
                nn = xp_df.iloc[u.get_nns_by_vector(timepoint[i][j], n_neighbors)]
                nn = Counter(nn.index).most_common(2)
                label, num = nn[0]
                yt.append(label)
            yp.append(yt)
        yp_all.append(yp)
    return yp_all

def main():
    parser = argparse.ArgumentParser()
    # data parameters
    parser.add_argument('--study', type=str, default="weinreb")
    parser.add_argument('-g', '--gpu', type=int, default=1)
    parser.add_argument('-p', '--perturb_genes', type=str)
    parser.add_argument('-l', '--z_score', type=float)
    parser.add_argument('-d', '--day', type=int, default=None)
    parser.add_argument('-c', '--cell_type', type=str, default=None)
    #simulation parameters
    parser.add_argument('--num_sims', type=int)
    parser.add_argument('--num_init', type=int)
    parser.add_argument('--num_steps', type=int)
    # outfile
    parser.add_argument('--out', type=str)
    args = parser.parse_args() 

    # torch parameters
    device = torch.device('cuda:{}'.format(args.gpu))

    # load model
    if args.study == "weinreb":
         std=torch.load("/data/gl/g2/sachit/data/klein/perturbations/std2.pt")
         ann_dir="/data/gl/g5/yhtgrace/workspace/sc-diffusions-beta/data/Klein2020_fate_smoothed/50_20_10.ann"
         base_dir="/data/gl/g5/yhtgrace/workspace/sc-diffusions-beta/experiments/weinreb-fate-train_batch-0.1/all_kegg-softplus_2_400-1e-06/"
         config_path = os.path.join(base_dir, 'seed_{}/config.pt'.format(1))
         config = SimpleNamespace(**torch.load(config_path))
         train_pt = base_dir + "seed_1/train.{}.pt".format("epoch_002500")
         checkpoint = torch.load(train_pt, map_location=device)
         kwargs = {}
         model = this.AutoGenerator(config)
         checkpoint = torch.load(train_pt, map_location=device)
         model.load_state_dict(checkpoint['model_state_dict'])
         model.to(device)
         time_elapsed = config.t
         # num_steps = int(np.round(time_elapsed / config.train_dt))
    if args.study == "veres":
         #pt=torch.load("/data/gl/g5/yhtgrace/workspace/sc-diffusions-beta/data/Veres2019/fate_train.pt")
         std=torch.load("/data/gl/g2/sachit/data/veres/perturbations/std.pt")
         ann_dir="/data/gl/g5/yhtgrace/workspace/sc-diffusions-beta/data/Veres2019/50_20_10.ann"
         base_dir="/data/gl/g5/yhtgrace/workspace/sc-diffusions-beta/experiments/veres-fate-all/kegg-softplus_2_400-1e-06/"
         config_path = os.path.join(base_dir, 'seed_{}/config.pt'.format(3))
         config = SimpleNamespace(**torch.load(config_path))

         train_pt = base_dir + "seed_3/train.{}.pt".format("epoch_001500")
         checkpoint = torch.load(train_pt, map_location=device)
         kwargs = {}
         model = this.AutoGenerator(config)
         checkpoint = torch.load(train_pt, map_location=device)
         model.load_state_dict(checkpoint['model_state_dict'])
         model.to(device)
         time_elapsed = config.t
         num_steps = int(np.round(time_elapsed / config.train_dt))


    # load data
    meta=std["meta"]
    xp_std = std["xp_std"]
    pca = std["pca_std"]

    message="Running perturbations of " + args.perturb_genes + " " + "w/ z-score of " + str(args.z_score)
    print(message)
    # induce perturbations
    xp = perturb(args, std, pca)
    
    # simulate 200 unperturbed and perturbed cells 1000 times
    unperturbed_sims, perturbed_sims  = simulate(args, device=device, model=model, config=config,  x_std=xp_std, x_perturb=xp, meta=meta)

    # classify cells for both outputs
    unperturbed_classes = classify_cells(args, unperturbed_sims, ann_dir, std)
    perturbed_classes = classify_cells(args, perturbed_sims, ann_dir, std)

    # write all files as torch
    if args.study=="weinreb":
        outfile="/data/gl/g2/sachit/data/klein/perturbations/"+args.out
    elif args.study=="veres":
        outfile="/data/gl/g2/sachit/data/veres/perturbations/"+args.out
    torch.save({"perturbed_genes": args.perturb_genes,
                "unperturbed_sim": unperturbed_sims,
                "unperturbed_labs": unperturbed_classes,
                "perturbed_sim": perturbed_sims,
                "perturbed_labs": perturbed_classes},
               outfile)
    
    
if __name__ == '__main__':
    main()
