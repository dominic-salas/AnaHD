"""The cleaning pipeline on an NVIDIA GPU (PyTorch, CUDA), same steps as the CPU version.

Needs the optional ``gpu`` extra (torch, torchvision). Optical flow is RAFT-small instead of
DIS; everything else follows the CPU steps in :mod:`anahd.clean` and :mod:`anahd.look`.
"""
