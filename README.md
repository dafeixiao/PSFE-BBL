# PSFE-BBL
This code is designed to build an imaging model of a photographic lens, with the help of a detection phase mask.

To test the code, please follow the steps below and install required python packages accordingly 

1, run tokina.ipynb to build the imaging model, save the model (checkpoint.pt in this code), and test the model on three unseen phase masks--empty phase, double helix, and another tetrapod.

2, run crlb_optim.ipynb to design a phase mask with optimal CRLB for 3D localization. Note that mask rotation, acompanied by PSF rotation, doesn't influence the CRLB.

The publication:

Dafei Xiao, Reut Orange Kedem, and Yoav Shechtman, "Point spread function modeling and engineering in black-box lens systems," Opt. Express 33, 4211-4224 (2025)
