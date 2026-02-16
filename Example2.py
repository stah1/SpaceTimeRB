#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a three-dimensional example with minimal regularity. To use the 
relative estimator, change the variable below.
"""
use_rel_estimator = False

import ufl
import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, mesh
from SpaceTimeRB import (build_spatial_matrices, build_spatial_vectors, 
build_temporal_matrices_CG1DG0, build_temporal_vectors_CG1DG0, build_S_q, 
build_s_q, hf_solve_exe, initialize_RB, offline_phase, build_W_inner_prod, 
prepare_fine_estimator, POD_greedy, save_space_time)



# Specify the reference parameter.
mu_ref = np.ones((6,1))
c_c = 0.25
c_s = 4



# Generating the spatial mesh.
gmsh.initialize()
dim = 3

left_box = gmsh.model.occ.addBox(0,0,0,0.5,1,0.5)
right_box = gmsh.model.occ.addBox(0.5,0.5,0,0.5,0.5,0.5)
body_tag = gmsh.model.occ.fuse([(dim, left_box)], [(dim, right_box)])
assert body_tag[0][0][0] == dim
body_tag = body_tag[0][0][1]

cyl1 = gmsh.model.occ.addCylinder(0.25,0.25,0,0,0,0.2,0.2)
cyl2 = gmsh.model.occ.addCylinder(0.25,0.75,0,0,0,0.2,0.2)
cyl3 = gmsh.model.occ.addCylinder(0.75,0.75,0,0,0,0.2,0.2)
res = gmsh.model.occ.fragment([(dim,body_tag)],
                              [(dim, cyl1),(dim, cyl2),(dim, cyl3)])

assert res[0][0][1] == cyl1
assert res[0][1][1] == cyl2
assert res[0][2][1] == cyl3
rest_tag = 5
assert res[0][3][1] == rest_tag
# These tags may differ depending on the version of Gmsh.
# In newer Gmsh versions it holds that res[0][1][1] == cyl1, 
# res[0][2][1] == cyl2, res[0][3][1] == cyl3 and res[0][0][1] == rest_tag

gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.07)

cyl1 = gmsh.model.addPhysicalGroup(dim, [cyl1])
cyl2 = gmsh.model.addPhysicalGroup(dim, [cyl2])
cyl3 = gmsh.model.addPhysicalGroup(dim, [cyl3])
rest_tag = gmsh.model.addPhysicalGroup(dim, [rest_tag])

gmsh.model.mesh.generate(dim)
gmsh.write("domain.mesh")
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(
    gmsh.model, mesh_comm, gmsh_model_rank, gdim=dim)

gmsh.finalize()



# Defining the function spaces and operators.
D = fem.functionspace(domain, ("DG", 0))
V = fem.functionspace(domain, ("CG", 1))

ind_c1 = fem.Function(D)
ind_c1.x.array[np.where(cell_markers.values == cyl1)[0]] = 1.0

ind_c2 = fem.Function(D)
ind_c2.x.array[np.where(cell_markers.values == cyl2)[0]] = 1.0

ind_c3 = fem.Function(D)
ind_c3.x.array[np.where(cell_markers.values == cyl3)[0]] = 1.0

dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_markers)

domain.topology.create_connectivity(dim-1, dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, dim-1, boundary_facets)

zero = fem.Function(V)
bc = fem.dirichletbc(zero, dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

operator_A = ufl.inner(ufl.grad(u), ufl.grad(v))
operators_A = [operator_A * dx(rest_tag),
               operator_A * dx(cyl1),
               operator_A * dx(cyl2),
               operator_A * dx(cyl3)]

operator_M = u*v*ufl.dx

theta_A_0 = lambda mu: 1.0
theta_A_1 = lambda mu: mu[0][0]
theta_A_2 = lambda mu: mu[1][0]
theta_A_3 = lambda mu: mu[2][0]

thetas_A = [theta_A_0, theta_A_1, theta_A_2, theta_A_3]
Q_A = len(thetas_A)

A_x, A_x_q, M_x, M_x_inv = build_spatial_matrices(mu_ref, thetas_A, 
                                                  operators_A, operator_M, bc)

assert len(operators_A) == Q_A
assert len(A_x_q) == Q_A

operators_f_x = [v.dx(0)*dx(cyl1), v.dx(0)*dx(cyl2), v.dx(0)*dx(cyl3)]

y_0 = fem.Function(V) # automatically 0
operators_y0 = [y_0 * v * ufl.dx]

operators_f_x, operators_y0 = build_spatial_vectors(
    mu_ref, thetas_A, operators_A, operators_f_x, operators_y0, bc)

thetas_f_x = [lambda mu: mu[3][0], lambda mu: mu[4][0], lambda mu: mu[5][0]]
assert len(operators_f_x) == len(thetas_f_x)
thetas_y0 = [lambda mu: 0.0*mu[0][0]+1.0]
assert len(operators_y0) == len(thetas_y0)

operators_A = A_x_q
del(A_x_q)



# Creating the temporal grid and operators.
t_0 = 0
t_n = 1
n_t = 15
t_arr = np.linspace(t_0, t_n, num=n_t+1) # t0 -- tn

A_t, M_t, T_t, Z_t, M_tD, M_tD_inv = build_temporal_matrices_CG1DG0(t_arr)

def f_t(t):
    if t <= 0.5:
        return 1.0
    else: 
        return 0.0
    
operators_f_t = [f_t]

R_0_t, [F_1_t], [F_2_t_SP], [F_2_t] = build_temporal_vectors_CG1DG0(
    t_arr, operators_f_t)
F_2_t = np.round(F_2_t, decimals=1) # Fix for numerical errors; okay here
# since we know that we have at most one decimal place.

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



# Generating the training set.
S_train = []
for mu_0 in np.geomspace(0.25, 4, num=10):
    for mu_1 in np.geomspace(0.25, 4, num=10):
        for mu_2 in np.geomspace(0.25, 4, num=10):
            for mu_3 in [1.0,2.0,3.0]:
                for mu_4 in [1.0,2.0,3.0]:
                    for mu_5 in [1.0,2.0,3.0]:
                        S_train.append(np.array([[mu_0],[mu_1],[mu_2],[mu_3],
                                                 [mu_4],[mu_5]]))



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
reduced_basis, _, _, _, _ = POD_greedy(
    mu_start, S_train, reduced_basis, Z, K_rb, G, thetas_A, thetas_A_mu_bar, 
    thetas_S_ref, S_q_ref, thetas_s_ref, s_q_ref, S_q_ref, S_q_rb, 
    thetas_S_ref, s_q_ref, s_q_rb, thetas_s_ref, hf_solve, M_x, M_x_inv, A_x, 
    operators_A, A_t, M_t, T_t, Z_t, M_tD_inv, inner_prod_part1, 
    inner_prod_part2, inner_prod_res, conv_tol=0, max_L=90, 
    use_rel_estimator=use_rel_estimator, A_x_q_solved=op_A_solved, 
    inner_prod_ref=inner_prod_ref)



# Export the first three basis functions for comparison.
for i in range(0, 3):
    save_space_time(reduced_basis[i], domain, t_arr, V, 
                    filename="basis " + str(i) + ".xdmf")