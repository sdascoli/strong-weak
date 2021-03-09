# More data or more parameters? Investigating the role of data structure on generalization"

This folder contains the SM as well as the codes to reproduce the experiments presented in the paper "More data or more parameters? Investigating the role of data structure on generalization", by Stéphane d'Ascoli, Marylou Gabrié, Levent Sagun and Giulio Biroli.

## Random features

The code for random feature models is in the ```random-features``` folder.
The data is pre-generated in the ```data``` folder, and can be plotted or regenerated using the interactive Jupyer notebooks.

## Neural networks

The code for neural networks is in the ```neural-networks``` folder.
Requires PyTorch>=1.5.0 and scikit-image==0.15.0.

To run a 2-layer network with 100 nodes on 10k MNIST examples downsampled to 10 by 10 images with 50% label corruption, run
``` python main.py --n 10000 --d 100 --depth 2 --width 100 --noise 0.5 ```

To reproduce the experiments presented in the paper, simply run submit.py (requires Slurm, and schedules around 1000 CPU-only jobs).
