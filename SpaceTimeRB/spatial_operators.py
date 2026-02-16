#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a module that contains methods for constructing system matrices and 
vectors with respect to space. The available methods are:\n
- build_spatial_matrices
- build_spatial_vectors
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from scipy.sparse import spdiags
from petsc4py import PETSc
from .high_fidelity import petsc_to_scipy




def build_spatial_matrices(mu_ref, thetas_A, operators_A, operator_M, bc, 
                           mass_lumping=True):
    """
    This method produces the parameter-decomposed components of the Riesz 
    isomorphism, A, in matrix form. It also generates a reference matrix, A_x, 
    for a given reference parameter, as well as a (lumped) mass matrix.

    Parameters
    ----------
    mu_ref : numpy.ndarray
        A reference parameter.
    thetas_A : list
        A list of functions that return theta_A^q(mu) for a given parameter mu.
    operators_A : list
        A list of objects of type ufl.form.Form that model the operators A_q.
    operator_M : ufl.form.Form
        An ufl form for the mass, in general u*v*ufl.dx.
    bc : dolfinx.fem.bcs.DirichletBC
        The Dirichlet boundary condition for the problem.
    mass_lumping : bool, optional
        Decides, whether mass lumping should be applied. The default is True.

    Returns
    -------
    A_x : scipy.sparse._csr.csr_array
        Matrix representation of the (reference) Riesz isomorphism A(mu_ref).
    A_x_q : list
        List of matrices A_x_q that represent the operators A_q.
    M_x : scipy.sparse._dia.dia_matrix or scipy.sparse._csr.csr_array
        Mass matrix of the problem, can be lumped.
    M_x_inv : scipy.sparse._dia.dia_matrix or NoneType
        Inverted mass matrix. This is only returned if mass lumping can be 
        applied.
    """
    
    mu_ref_A = np.array([theta(mu_ref) for theta in thetas_A])
    
    operator_A_ref = mu_ref_A.dot(operators_A)    
    operator_A_cpp = fem.form(operator_A_ref)
    operator_A_mat = fem.petsc.assemble_matrix(operator_A_cpp, bcs=[bc])
    operator_A_mat.assemble()
    A_x = petsc_to_scipy(operator_A_mat)
    
    A_x_q = []
    for op in operators_A:
        operator_A_cpp = fem.form(op)
        operator_A_mat = fem.petsc.assemble_matrix(operator_A_cpp, bcs=[bc])
        operator_A_mat.assemble()
        A_x_q.append(petsc_to_scipy(operator_A_mat))
        
    mass_cpp = fem.form(operator_M)
    mass_mat = fem.petsc.assemble_matrix(mass_cpp, bcs=[bc])
    mass_mat.assemble()
    M_x = petsc_to_scipy(mass_mat)
    
    if mass_lumping:
        M_x = spdiags([M_x.sum(axis=1)], [0])
        M_x_inv = spdiags([1./M_x.diagonal()], [0])
        
        return A_x, A_x_q, M_x, M_x_inv
    else:
        return A_x, A_x_q, M_x, None




def build_spatial_vectors(mu_ref, thetas_A, operators_A, operators_f, 
                          operators_y0, bc):
    """
    This method produces parameter-decomposed vectors for use in the right-hand
    side later on.

    Parameters
    ----------
    mu_ref : numpy.ndarray
        A reference parameter.
    thetas_A : list
        A list of functions that return theta_A^q(mu) for a given parameter mu.
    operators_A : list
        A list of objects of type ufl.form.Form that model the operators A_q.
    operators_f : list
        A list of objects of type ufl.form.Form that model the operators f with
        respect to space.
    operators_y0 : list
        A list of objects of type ufl.form.Form that model the initial 
        condition.
    bc : dolfinx.fem.bcs.DirichletBC
        The Dirichlet boundary condition for the problem.

    Returns
    -------
    F_x_q : list
        A list of numpy.ndarray vectors representing the (spatial) action of f 
        on a test function.
    R_0_x_q : list
        A list of numpy.ndarray vectors representing the action of (y_0, .)_H
        on a (spatial) test function.
    """
    
    mu_ref_A = np.array([theta(mu_ref) for theta in thetas_A])
    operator_A_ref = mu_ref_A.dot(operators_A)    
    operator_A_cpp = fem.form(operator_A_ref)
    
    F_x_q = []
    for op in operators_f:
        F_x_vec = fem.petsc.assemble_vector(fem.form(op))
        F_x_vec.assemble()
        fem.petsc.apply_lifting(F_x_vec, [operator_A_cpp], [[bc]])
        F_x_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                            mode=PETSc.ScatterMode.REVERSE) 
        fem.petsc.set_bc(F_x_vec, [bc])
        F_x = np.array([F_x_vec.array]).T
        F_x_q.append(F_x)
        
    R_0_x_q = []
    for op in operators_y0:
        R_0_cpp = fem.form(op)
        R_0_vec = fem.petsc.assemble_vector(R_0_cpp)
        R_0_vec.assemble()
        fem.petsc.apply_lifting(R_0_vec, [operator_A_cpp], [[bc]])
        R_0_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                            mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(R_0_vec, [bc])
        R_0_x = np.array([R_0_vec.array]).T
        R_0_x_q.append(R_0_x)
    
    return F_x_q, R_0_x_q