#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a standard 2D thermal block example to compare the error estimators.
"""

import ufl
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from SpaceTimeRB import (build_spatial_matrices, build_spatial_vectors, 
build_temporal_matrices_CG1DG0, build_temporal_vectors_CG1DG0, build_S_q, 
build_s_q, hf_solve_exe, initialize_RB, offline_phase, build_W_inner_prod, 
prepare_fine_estimator, POD_greedy)



# Specify the reference parameter.
mu_ref = np.ones((9,1))
c_c = 0.1
c_s = 10.0



# Generating the spatial domain and operators.
mesh_comm = MPI.COMM_WORLD
n_x = 21
domain = mesh.create_unit_square(comm=mesh_comm, nx=n_x, ny=n_x)
V = fem.functionspace(domain, ("CG", 1))
D = fem.functionspace(domain, ("DG", 0))

def on_boundary(x):
    ret = []            
    for i in range(0, x.shape[1]):
        if np.isclose(x[1][i], 1):
            ret.append(1.0)
        else:
            ret.append(0.0)
    return np.array(ret)


dofs = fem.locate_dofs_geometrical(V, on_boundary)
zero = fem.Function(V)
bc = fem.dirichletbc(zero, dofs)

def check_point(k, x): # For the definition of the thermal block.
    assert k in range(1,10)
    line = (k-1)//3 + 1
    col =  (k-1)%3 + 1
    ret = []
    for i in range(0, x.shape[1]):
        if x[0][i] >= (col-1)/3 and x[0][i] <= col/3\
            and x[1][i] >= (line-1)/3 and x[1][i] <= line/3:
            ret.append(1.0)
        else:
            ret.append(0.0)
    return np.array(ret)

indicators = []
for k in range(1,10):
    _ind_here = fem.Function(D)
    _ind_here.interpolate(lambda x: check_point(k, x))
    indicators.append(_ind_here.copy())

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

operator_A = ufl.inner(ufl.grad(u), ufl.grad(v))
operators_A = []
for k in range(1,10):
    operators_A.append(operator_A * indicators[k-1] * ufl.dx)

operator_M = u*v*ufl.dx

thetas_A = []
for k in range(0,9):
    if k != 8:
        ind = k
        thetas_A.append(lambda mu: mu[ind][0])
    else:
        thetas_A.append(lambda mu: 0.0*mu[0][0] + 1.0)
Q_A = len(thetas_A)

A_x, A_x_q, M_x, M_x_inv = build_spatial_matrices(mu_ref, thetas_A, 
                                                  operators_A, operator_M, bc)

assert len(operators_A) == Q_A
assert len(A_x_q) == Q_A

def on_bottom(x):
    ret = []            
    for i in range(0, x.shape[1]):
        if np.isclose(x[1][i], 0):
            ret.append(1.0)
        else:
            ret.append(0.0)
    return np.array(ret)

facets_bottom = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, 
                                              on_bottom)
facet_tag_value = 999
tags = np.full(len(facets_bottom), facet_tag_value, dtype=np.int32)
facet_mt = mesh.meshtags(domain, domain.topology.dim - 1, facets_bottom, tags)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_mt)
operators_f_x = [v * ds(facet_tag_value)]

y_0 = fem.Function(V) # automatically 0
operators_y0 = [y_0 * v * ufl.dx]

operators_f_x, operators_y0 = build_spatial_vectors(
    mu_ref, thetas_A, operators_A, operators_f_x, operators_y0, bc)

thetas_f_x = [lambda mu: mu[8][0]]
thetas_y0 = [lambda mu: 0.0*mu[0][0]+1.0]

operators_A = A_x_q
del(A_x_q)



# Creating the temporal grid and operators.
t_0 = 0
t_n = 3 
n_t = 60
t_arr = np.linspace(t_0, t_n, num=n_t+1)

A_t, M_t, T_t, Z_t, M_tD, M_tD_inv = build_temporal_matrices_CG1DG0(t_arr)

def f_t(t):
    return 1.0

operators_f_t = [f_t]

R_0_t, [F_1_t], [F_2_t_SP], [F_2_t] = build_temporal_vectors_CG1DG0(
    t_arr, operators_f_t)
F_2_t[1:-1] = 0.0 # Bugfix for numerical errors; works since we have f_t = 1

thetas_f_t = [lambda mu: 0.0*mu[0][0]+1.0]



# Building the offline-online-decomposition of the saddle point problem.
S_q, thetas_S = build_S_q(M_x, thetas_A, operators_A, M_t, T_t, M_tD=M_tD, 
                          Z_t=Z_t)
s_q, thetas_s = build_s_q(thetas_A, operators_A, thetas_f_x, operators_f_x, 
                          thetas_y0, operators_y0, thetas_f_t, [F_1_t], 
                          [F_2_t_SP], R_0_t)

# Useful matrices and vectors for later use.
S_q_ref, thetas_S_ref = build_S_q(M_x, thetas_A, operators_A, M_t, T_t, 
                                  M_x_inv=M_x_inv, A_t=A_t, 
                                  use_reformulation=True)
s_q_ref, thetas_s_ref = build_s_q(thetas_A, operators_A, thetas_f_x, 
                                  operators_f_x, thetas_y0, operators_y0, 
                                  thetas_f_t, [F_1_t], [F_2_t], R_0_t, 
                                  M_x_inv=M_x_inv, use_reformulation=True)



# Define a method for the high fidelity solve.
def hf_solve(mu):
    return hf_solve_exe(mu, thetas_S, S_q, thetas_s, s_q, comm=domain.comm, 
                        A_t=A_t, M_t=M_t, Z_t=Z_t, M_tD=M_tD, M_x=M_x, 
                        thetas_A=thetas_A, A_x_q=operators_A)



# Generating the training and validation set.
np.random.seed(31565)
S_train = 2*(np.random.rand(9,5000)-0.5)
S_train[:-1,:] = 10**S_train[:-1,:]
S_train = list(np.array([S_train]).T)

np.random.seed(22959)
mu_validation = 2*(np.random.rand(9,100)-0.5)
mu_validation[:-1,:] = 10**mu_validation[:-1,:]
mu_validation = list(np.array([mu_validation]).T)

y_validation = []
for mu_val_loc in mu_validation:
    print("Solving...")
    y_validation.append(hf_solve(mu_val_loc))



# Defining the start parameter. 
mu_start = S_train[int(len(S_train)/2)]
S_train[int(len(S_train)/2)] = None



# Initialize RB.
inner_prod_part1, inner_prod_part2, inner_prod_res, reduced_basis, \
    reduced_basis_Riesz, Z, K_rb, thetas_A_mu_bar = initialize_RB(
        mu_ref, mu_start, thetas_A, A_t, M_t, T_t, Z_t, M_tD_inv, M_x, M_x_inv, 
        A_x, hf_solve)
G, S_q_rb, s_q_rb = offline_phase(reduced_basis, S_q_ref, s_q_ref, 
                                  inner_prod_res)
inner_prod_ref = build_W_inner_prod(A_x, M_x, A_t, M_t, T_t)
op_A_solved = prepare_fine_estimator(A_x, operators_A)



# Run the POD-greedy algorithm.
POD_greedy(mu_start, S_train, reduced_basis, Z, K_rb, G, thetas_A, 
           thetas_A_mu_bar, thetas_S_ref, S_q_ref, thetas_s_ref, s_q_ref, 
           S_q_ref, S_q_rb, thetas_S_ref, s_q_ref, s_q_rb, thetas_s_ref, 
           hf_solve, M_x, M_x_inv, A_x, operators_A, A_t, M_t, T_t, Z_t, 
           M_tD_inv, inner_prod_part1, inner_prod_part2, inner_prod_res,
           conv_tol=0, A_x_q_solved = op_A_solved, mu_val_set=mu_validation, 
           y_val_set=y_validation, inner_prod_ref=inner_prod_ref)