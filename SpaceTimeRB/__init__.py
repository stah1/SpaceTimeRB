#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a library containing modules for applying reduced basis methods to our 
least squares space-time approach for parabolic equations, which assumes only 
minimal regularity. 

Some proof-of-concept methods are also included for research purposes.

We recommend reading the article and installation instructions before use.

The available methods are:\n
- build_spatial_matrices
- build_spatial_vectors
- CG1_1D (basis function)
- DG0_1D (basis function)
- gauss_quadrature
- create_mass_CG1
- create_stiffness_CG1
- create_mass_DG0
- create_transport_CG1_DG0
- build_temporal_matrices_CG1DG0
- build_temporal_vectors_CG1DG0
- save_space_time
- scipy_to_petsc
- petsc_to_scipy
- remove_dofs
- solve_petsc
- preconditioner
- assemble_mat_mu
- assemble_vec_mu
- hf_solve_exe
- prepare_correlation
- build_W_inner_prod
- initialize_RB
- POD
- build_S_q
- build_s_q
- offline_phase
- rb_solve
- min_theta
- max_theta
- estimator_part1
- estimator_part2
- prepare_fine_estimator
- fine_estimator
- POD_greedy
"""

from .spatial_operators import *
from .temporal_operators import *
from .high_fidelity import *
from .offline_phase import *
from .online_phase import *