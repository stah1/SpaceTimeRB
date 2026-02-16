#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains methods that only need to be run during the offline phase. 
The available methods are:\n
- initialize_RB
- POD
- build_S_q
- build_s_q
- offline_phase
"""

from .high_fidelity import prepare_correlation
from scipy.sparse import kron, bmat, csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np




def initialize_RB(mu_ref, mu_start, thetas_A, A_t, M_t, T_t, Z_t, M_tD_inv, 
                  M_x, M_x_inv, A_x, hf_solve):
    """
    This method initializes the reduced basis approach and computes all the 
    necessary objects for the start.

    Parameters
    ----------
    mu_ref : numpy.ndarray
        The reference parameter.
    mu_start : numpy.ndarray
        A start parameter to solve for.
    thetas_A : list
        A list of functions that return theta_A^q(mu) for a given parameter mu.
    A_t : scipy.sparse._dia.dia_matrix
        The stiffness matrix in time for CG1 elements.
    M_t : scipy.sparse._dia.dia_matrix
        The mass matrix in time for CG1 elements.
    T_t : scipy.sparse._coo.coo_array
        The matrix \chi_j(T)\chi_i(T) for CG1 elements \chi.
    Z_t : scipy.sparse._dia.dia_matrix
        The CG1-DG0 transport matrix.
    M_tD_inv : scipy.sparse._dia.dia_matrix
        The inverse of M_tD (which is diagonal).
    M_x : scipy.sparse._dia.dia_matrix
        Mass matrix with respect to space.
    M_x_inv : scipy.sparse._dia.dia_matrix
        Inverted mass matrix.
    A_x : scipy.sparse._csr.csr_array
        The matrix representation of A for the reference parameter.
    hf_solve : function
        A local function that returns the solution for a given parameter. This 
        function should be based on hf_solve_exe.

    Returns
    -------
    inner_prod_part1 : scipy.sparse._csr.csr_array
        First part of the W_d inner product.
    inner_prod_part2 : scipy.sparse._coo.coo_matrix
        Second part of the W_d inner product.
    inner_prod_res : scipy.sparse._csr.csr_array
        Matrix for computing the residual norm estimation.
    reduced_basis : list
        List of DOF vectors of the reduced basis for y.
    reduced_basis_Riesz : list
        List of DOF vectors of the Riesz representer of the reduced basis.
    Z : list
        Collection of all computed high-fidelity solutions and their Riesz 
        representers in the form of a list of tuples.
    K_rb : numpy.ndarray
        Matrix representation of the inner product in the reduced basis space.
    thetas_A_mu_bar : list
        List of evaluated parameter functions theta_A^q for the reference 
        parameter.
    """
    
    inner_prod_part1 = kron(M_t, A_x) + kron(T_t, M_x)
    inner_prod_part2 = kron(Z_t.T, M_x)
    
    inner_prod_res = kron(A_t, A_x) \
        + kron(M_t, A_x @ M_x_inv @ A_x @ M_x_inv @ A_x) \
        + kron(T_t, A_x @ M_x_inv @ A_x)

    print("Solving for the first high fidelity solution.")
    y_start = hf_solve(mu_start)

    y_start_prep = prepare_correlation(y_start, M_x, A_x, Z_t, M_tD_inv)
    y_start_norm = (y_start.T @ inner_prod_part1 @ y_start)[0,0]
    y_start_norm += (y_start.T @ inner_prod_part2 @ y_start_prep)[0,0]
    y_start_norm = np.sqrt(y_start_norm)

    reduced_basis = [y_start/y_start_norm]
    reduced_basis_Riesz = [y_start_prep/y_start_norm]
    Z = [(y_start, y_start_prep)]
    K_rb = np.array([[1.0]])
    
    thetas_A_mu_bar = []
    for i in range(0, len(thetas_A)):
        thetas_A_mu_bar.append(thetas_A[i](mu_ref))
    
    return inner_prod_part1, inner_prod_part2, inner_prod_res, reduced_basis,\
        reduced_basis_Riesz, Z, K_rb, thetas_A_mu_bar




def POD(snapshots, N_rb, inner_prod_part1, inner_prod_part2):
    """
    Performs POD on a given set of snapshots.

    Parameters
    ----------
    snapshots : list
        Collection of all computed high-fidelity solutions and their Riesz 
        representers in the form of a list of tuples.
    N_rb : int
        Desired reduced basis space dimension.
    inner_prod_part1 : scipy.sparse._csr.csr_array
        The first part of the W_d inner product from initialize_RB.
    inner_prod_part2 : scipy.sparse._coo.coo_matrix
        Second part of the W_d inner product from initialize_RB.

    Returns
    -------
    reduced_basis : list
        List of DOF vectors of the basis functions.
    K_rb : numpy.ndarray
        Matrix representation of the inner product in the reduced basis space.
    reduced_basis_Riesz : list
        List of DOF vectors of the Riesz representer of the reduced basis.
    """

    L = len(snapshots)
    corr = np.zeros((L, L))
    for i in range(L):
        corr[i, i] = snapshots[i][0].T @ inner_prod_part1 @ snapshots[i][0]
        corr[i, i] +=snapshots[i][0].T @ inner_prod_part2 @ snapshots[i][1]
        for j in range(i):
            corr[i, j] = snapshots[i][0].T @ inner_prod_part1 @ snapshots[j][0]
            corr[i, j] +=snapshots[i][0].T @ inner_prod_part2 @ snapshots[j][1]
    corr = corr + corr.T - np.diag(np.diag(corr))
    
    corr = 1.0/L * corr
    eigenvals, eigenvectors = np.linalg.eigh(corr)
    
    new_rb = []
    new_rb_Riesz = []
    
    max_nodes = min(L, N_rb)
    for i in range(max_nodes):
        j = L-1-i
        vec_loc = eigenvectors[:, j]
        
        rb_loc = np.zeros((snapshots[0][0].size, 1))
        rb_loc_Riesz = np.zeros((snapshots[0][1].size, 1))
        
        for k in range(L):
            rb_loc += (1.0/np.sqrt(L))*vec_loc[k]*snapshots[k][0]
            rb_loc_Riesz += (1.0/np.sqrt(L))*vec_loc[k]*snapshots[k][1]
            
        new_rb.append(rb_loc)
        new_rb_Riesz.append(rb_loc_Riesz)
        
    eigenvals = eigenvals[L-max_nodes:]
    eigenvals = np.flip(eigenvals)
    return new_rb, np.diag(eigenvals), new_rb_Riesz




def build_S_q(M_x, thetas_A, A_x_q, M_t, T_t, use_reformulation=False,
              M_x_inv = None, A_t=None, M_tD=None, Z_t=None):
    """
    This method assembles the space-time matrices S_q for the use in the high 
    fidelity and reduced problem formulations.

    Parameters
    ----------
    M_x : scipy.sparse._dia.dia_matrix
        Mass matrix with respect to space.
    thetas_A : list
        A list of parameter functions to assemble A(\mu).
    A_x_q : list
        A list of matrices representing A_q.
    M_t : scipy.sparse._dia.dia_matrix
        The mass matrix in time.
    T_t : scipy.sparse._coo.coo_array
        The matrix \chi_j(T)\chi_i(T) for CG1 elements \chi.
    use_reformulation : bool, optional
        Needs to be set to True, if the saddle point problem is reduced to an 
        equation for y only. The default is False.
    M_x_inv : scipy.sparse._dia.dia_matrix, optional
        Inverted mass matrix. This matrix only needs to be provided, if the 
        reformulation is used. The default is None.
    A_t : scipy.sparse._dia.dia_matrix, optional
        The stiffness matrix in time. This is only needed if the reformulation 
        is used. The default is None.
    M_tD : scipy.sparse._dia.dia_matrix, optional
        The DG0 mass matrix in time. This is not needed if the reformulation 
        is used. The default is None.
    Z_t : scipy.sparse._dia.dia_matrix, optional
        The CG1-DG0 transport matrix in time. This is not needed if the 
        reformulation is used. The default is None.

    Returns
    -------
    S_q : list
        A list of matrices S_q.
    thetas_S : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent system matrix S(\mu).
    """
    
    assert not use_reformulation or M_x_inv is not None
    assert not use_reformulation or A_t is not None
    assert use_reformulation or M_tD is not None
    assert use_reformulation or Z_t is not None
    
    Q_A = len(A_x_q)
    S_q = []
    
    if use_reformulation:
        S_q.append(kron(A_t, M_x))
        for j in range(0, Q_A):
            for i in range(0, Q_A):            
                S_q.append(kron(M_t, A_x_q[i] @ M_x_inv @ A_x_q[j]))
        for i in range(0, Q_A):
            S_q.append(kron(T_t, A_x_q[i])) 
            
        def thetas_S(mu):
            ret = [1.0]
            for j in range(0, Q_A):
                for i in range(0, Q_A):
                    ret.append(thetas_A[i](mu)*thetas_A[j](mu))
            for i in range(0, Q_A):
                ret.append(thetas_A[i](mu))
            return ret
        
        return S_q, thetas_S
    
    else:
        for i in range(0, Q_A):
            S_q.append(bmat([[kron(M_tD, A_x_q[i]), None], 
                                             [None, -kron(M_t, A_x_q[i])]]))
        S_q.append(bmat([[None, -kron(Z_t, M_x)], 
                         [-kron(Z_t.T, M_x), -kron(T_t, M_x)]]))

        def thetas_S(mu):
            ret = []
            for i in range(0, Q_A):
                ret.append(thetas_A[i](mu))
            ret.append(1.0)
            return ret
        
        return S_q, thetas_S




def build_s_q(thetas_A, A_x_q, thetas_f_x, F_x_q, thetas_y0, R_0_x_q, 
              thetas_f_t, F_1_t_q, F_2_t_q, R_0_t, use_reformulation=False, 
              M_x_inv=None):
    """
    This method assembles the space-time vectors s_q for the use in the high 
    fidelity and reduced problem formulations.

    Parameters
    ----------
    thetas_A : list
        A list of parameter functions to assemble A(\mu).
    A_x_q : list
        A list of matrices representing A_q.
    thetas_f_x : list
        A list of functions returning the parameter-dependent coefficients for 
        the spatial part of f_q.
    F_x_q : list
        A list of vectors representing the action of the spatial part f^x of 
        f = f^tf^x on the test functions.
    thetas_y0 : list
        A list of functions returning the parameter-dependent coefficients for 
        y_0.
    R_0_x_q : list
        A list of vectors representing the action of y_{0,q} on the spatial 
        test functions.
    thetas_f_t : list
        A list of functions returning the parameter-dependent coefficients for 
        the temporal part of f_q.
    F_1_t_q : list
        A list of numpy.ndarray vectors of the form \int f^t_q\chi_j, where f^t 
        is the temporal part of f from L^2(0,T) and \chi_j is a CG1 basis 
        function.
    F_2_t_q : list
        A list of numpy.ndarray vectors of the form \int f^t_q\psi_j, where f^t 
        is the temporal part of f from L^2(0,T) and \psi_j is a DG0 basis 
        function. If the reformulation is used this becomes a list of 
        numpy.ndarray vectors of the form \int f^t_q(\chi_j)_t, where \chi_j is 
        a CG1 basis function.
    R_0_t : numpy.ndarray
        A vector of the form \chi_j(0) for CG1 elements \chi.
    use_reformulation : bool, optional
        Needs to be set to True, if the saddle point problem is reduced to an 
        equation for y only. The default is False.
    M_x_inv : scipy.sparse._dia.dia_matrix, optional
        Inverted mass matrix. This matrix only needs to be provided, if the 
        reformulation is used. The default is None.

    Returns
    -------
    s_q : list
        A list of vectors s_q.
    thetas_s : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent vector s(\mu).
    """
    
    assert not use_reformulation or M_x_inv is not None
    
    Q_A = len(A_x_q)
    Q_y = len(R_0_x_q)
    Q_t = len(F_1_t_q)
    Q_x = len(F_x_q)
    s_q = []
    
    if use_reformulation:
        for i in range(0, Q_A):
            for j in range(0, Q_y):
                s_q.append(kron(R_0_t, A_x_q[i] @ M_x_inv @ R_0_x_q[j]))
    
        for i in range(0, Q_t):
            for j in range(0, Q_A):
                for k in range(0, Q_x):
                    s_q.append(kron(F_1_t_q[i], A_x_q[j] @ M_x_inv @ F_x_q[k]))
    
        for i in range(0, Q_t):
            for j in range(0, Q_x):
                s_q.append(kron(F_2_t_q[i], F_x_q[j]))
    
        def thetas_s(mu):
            ret = []
            for i in range(0, Q_A):
                for j in range(0, Q_y):
                    ret.append(thetas_A[i](mu)*thetas_y0[j](mu))
            for i in range(0, Q_t):
                for j in range(0, Q_A):
                    for k in range(0, Q_x):
                        ret.append(thetas_f_t[i](mu)*thetas_A[j](mu)\
                                   *thetas_f_x[k](mu))
            for i in range(0, Q_t):
                for j in range(0, Q_x):       
                    ret.append(thetas_f_t[i](mu)*thetas_f_x[j](mu))
            return ret
    
    else:
        for i in range(0, Q_t):
            for j in range(0, Q_x):
                s_q.append(bmat([[-kron(F_2_t_q[i], F_x_q[j])],
                                 [-kron(F_1_t_q[i], F_x_q[j])]]))
                
        for i in range(0, Q_y):
            s_q.append(bmat([[csr_matrix((F_2_t_q[0].shape[0]\
                                          *F_x_q[0].shape[0],1))],
                             [-kron(R_0_t, R_0_x_q[i])]]))

        def thetas_s(mu):
            ret = []
            for i in range(0, Q_t):
                for j in range(0, Q_x):
                    ret.append(thetas_f_t[i](mu)*thetas_f_x[j](mu))
            for i in range(0, Q_y):
                ret.append(thetas_y0[i](mu))
            return ret

    for k in range(len(s_q)):
        s_q[k] = s_q[k].toarray()
    
    return s_q, thetas_s




def offline_phase(reduced_basis, S_q, s_q, inner_prod_res, 
                  use_reformulation=True, S_q_reformulation=None, 
                  s_q_reformulation=None, reduced_basis_Riesz=None):
    """
    The offline phase must be run whenever a new reduced basis space becomes 
    available. This method produces the necessary reduced matrices and vectors.

    Parameters
    ----------
    reduced_basis : list
        List of DOF vectors of the basis functions.
    S_q : list
        A list of parameter-independent high fidelity matrices S_q.
    s_q : list
        A list of parameter-independent high fidelity vectors s_q.
    inner_prod_res : scipy.sparse._csr.csr_array
        Matrix for computing the residual norm estimation.
    use_reformulation : bool, optional
        Needs to be set to True, if the saddle point problem has been reduced 
        to an equation for y only. The default is True.
    S_q_reformulation : list, optional
        List of parameter-independent space-time system matrices S_q for the 
        reformulation, where the saddle point problem is reduced to an equation 
        for y only. These matrices need to be provided if no reformulation is 
        used. The default is None.
    s_q_reformulation : list, optional
        List of parameter-independent space-time right-hand side vectors s_q of 
        the reformulation. These vectors need to be provided if no 
        reformulation is used. The default is None.
    reduced_basis_Riesz : list, optional
        List of DOF vectors of the Riesz representer of the reduced basis. If 
        the reformulation is used, this is not needed. The default is None.

    Returns
    -------
    G : numpy.ndarray
        Matrix for computing the residual estimation in the reduced basis 
        space.
    S_q_rb : list
        The reduced system matrices.
    s_q_rb : list
        The reduced right-hand side vectors.
    """
    
    assert use_reformulation or (S_q_reformulation is not None and \
                                 s_q_reformulation is not None and \
                                     reduced_basis_Riesz is not None)
    
    if use_reformulation:
        S_q_reformulation = S_q
        s_q_reformulation = s_q
    
    B = np.array(reduced_basis).T[0]
    R = np.zeros((reduced_basis[0].shape[0], len(s_q_reformulation)\
                  +len(S_q_reformulation)*len(reduced_basis))) 
    
    for i in range(0, len(s_q_reformulation)):
        R[:, i] = s_q_reformulation[i].T
    index = len(s_q_reformulation)
    for i in range(0, len(S_q_reformulation)):
        R[:, index:index+len(reduced_basis)] = S_q_reformulation[i] @ B
        index += len(reduced_basis)
    
    G = R.T @ spsolve(inner_prod_res, R)

    S_q_rb = []
    s_q_rb = []
    
    if not use_reformulation:
        B_Riesz = np.array(reduced_basis_Riesz).T[0]
        B_sp = bmat([[B_Riesz,None],[None,B]])
        B = B_sp

    for i in range(0, len(S_q)):
        if use_reformulation:
            S_q_rb.append((B.T @ S_q[i] @ B))
        else:
            S_q_rb.append((B.T @ S_q[i] @ B).toarray())
    for i in range(0, len(s_q)):
        s_q_rb.append(B.T @ s_q[i])

    return G, S_q_rb, s_q_rb