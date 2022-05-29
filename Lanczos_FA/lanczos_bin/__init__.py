#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg



from .barycentric import compute_barycentric_weights, barycentric
from .lanczos import exact_lanczos,polyval_A_equil,lanczos_poly_approx,lanczos_CG_residual_coeff,lanczos_FA,lanczos_fAb,opt_poly_approx,opt_FA,opt_fAb,Q_wz,Q_z,get_a_priori_bound,get_a_posteriori_bound,get_exact_bound
from .remez import check_reference,remez,get_initial_reference
from .misc import get_discrete_nodes,get_cheb_nodes,model_problem_spectrum,discrete_laplacian_spectrum