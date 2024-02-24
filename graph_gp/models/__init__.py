# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.

# from .exact_swwlgp import BaseSWWLGP, SWWLGP, GPySWWLGP
from .classifier import SVC_precomputed_distances, SVC_precomputed_Gram
from .kernel import custom_kernel_exp, custom_kernel_just_scalars, custom_kernel_rbf
from .regressor import RGASP

__all__ = [
    "SVC_precomputed_distances",
    "SVC_precomputed_Gram",
    "custom_kernel_exp",
    "custom_kernel_rbf",
    "custom_kernel_just_scalars",
    "RGASP",
]
