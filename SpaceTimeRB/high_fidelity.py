#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains all the methods that operate at the high-fidelity level. 
The high-fidelity saddle-point formulation and solution of the problem are 
obtained here, in particular. The available methods are:\n
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
"""

import numpy as np
from scipy.sparse import kron, csr_matrix, csr_array, issparse
from scipy.sparse.linalg import spsolve, LinearOperator, gmres
from scipy.linalg import eigh
from dolfinx import fem, io
from petsc4py import PETSc




def save_space_time(y, domain, t_arr, V, filename="function.xdmf"):
    """
    Exports a time trajectory to an xdmf file.

    Parameters
    ----------
    y : numpy.ndarray
        The space-time DOF vector of the function to export.
    domain : dolfinx.mesh.Mesh
        Spatial domain.
    t_arr : numpy.ndarray
        Time grid.
    V : dolfinx.fem.function.FunctionSpace
        Spatial function space of y (should be CG 1).
    filename : str, optional
        Filename. The default is "function.xdmf".

    Returns
    -------
    None.
    """
    
    n_t = t_arr.size-1
    xdmf = io.XDMFFile(domain.comm, filename, "w")
    xdmf.write_mesh(domain)
    jump = int(y.size/(n_t+1))
    for k in range(0, n_t+1):
        y_func = fem.Function(V)
        y_func.x.array[:] = (y[(k*jump):((k+1)*jump)]).T[0]
        xdmf.write_function(y_func.copy(), t_arr[k])
    xdmf.close() 




def scipy_to_petsc(A):
    """
    Converts a SciPy matrix into PETSc format.

    Parameters
    ----------
    A : scipy.sparse matrix
        The matrix to convert.

    Returns
    -------
    petsc_mat : petsc4py.PETSc.Mat
        The converted matrix.
    """
    
    A = A.tocsr()
    petsc_mat = PETSc.Mat().createAIJ(
        size=A.shape,
        csr=(A.indptr, A.indices, A.data),
        comm=PETSc.COMM_WORLD
    )
    petsc_mat.assemble()
    return petsc_mat




def petsc_to_scipy(mat):
    """
    Converts a PETSc matrix into SciPy format.

    Parameters
    ----------
    mat : petsc4py.PETSc.Mat
        The matrix to convert.

    Returns
    -------
    A : scipy.sparse._csr.csr_array
        The converted matrix.
    """
    
    row_indices, col_indices, data = mat.getValuesCSR()
    return csr_array((data, col_indices, row_indices), shape=mat.getSize())




def remove_dofs(M, dofs):
    """
    This method removes rows and columns of a matrix M at the given degrees of 
    freedom (DOFs). This method can be used to remove boundary DOFs, for 
    example.

    Parameters
    ----------
    M : scipy.sparse._csr.csr_array
        The matrix from which the rows and columns are to be removed.
    dofs : numpy.ndarray
        An array of DOFs.

    Returns
    -------
    M : scipy.sparse._csr.csr_array
        The new matrix, reduced in its size by the removed rows and columns.
    """
    
    M = M.tocsr()
    mask = np.ones(M.shape[1], dtype=bool)
    mask[dofs] = False
    M = M[:, mask]
    M = M[mask, :]
    return M




def solve_petsc(A, b, sol, comm):
    """
    Solves A @ sol = b using PETSc. This method is used in the preconditioner.

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        An spd matrix.
    b : petsc4py.PETSc.Vec
        The right-hand side vector.
    sol : petsc4py.PETSc.Vec
        A PETSc vector to store the solution in.
    comm : mpi4py.MPI.Intracomm
        The Intracomm for PETSc.

    Returns
    -------
    None.
    """
    
    A.setOption(PETSc.Mat.Option.SPD, True)
    solver = PETSc.KSP().create(comm)
    solver.setFromOptions()
    solver.setOperators(A)
    solver.solve(b, sol)
    solver.destroy()




def preconditioner(b, N, M, W, Lam_t, A_petsc, M_petsc, M_x, Z_t, M_tD, comm):
    """
    The preconditioner for the perturbed saddle point problem.

    Parameters
    ----------
    b : numpy.ndarray
        The right-hand side to solve for.
    N : int
        Number of DOFs with respect to space.
    M : int
        Number of DOFs with respect to time (for CG 1).
    W : numpy.ndarray
        Matrix of eigenvectors of the generalized eigenvalue problem 
        W.T @ A_t @ W = diag(Lam_t), W.T @ M_t @ W = I_t, where we use that 
        Z_t.T @ M_tD_inv @ Z_t = A_t.
    Lam_t : numpy.ndarray
        Array of eigenvalues of the generalized eigenvalue problem 
        W.T @ A_t @ W = diag(Lam_t), W.T @ M_t @ W = I_t, where we use that 
        Z_t.T @ M_tD_inv @ Z_t = A_t.
    A_petsc : petsc4py.PETSc.Mat
        Matrix representation of A(mu) in PETSc format.
    M_petsc : petsc4py.PETSc.Mat
        Mass matrix with respect to space in PETSc format.
    M_x : scipy.sparse._dia.dia_matrix
        Mass matrix with respect to space.
    Z_t : scipy.sparse._dia.dia_matrix
        CG1-DG0 transport matrix in time.
    M_tD : scipy.sparse._dia.dia_matrix
        Mass matrix in time with DG0 elements.
    comm : mpi4py.MPI.Intracomm
        The Intracomm for PETSc.

    Returns
    -------
    sol : numpy.ndarray
        The solution of the preconditioner.
    """
    
    part_R = b[:(M-1)*N]
    part_Y = b[(M-1)*N:]

    part_R = part_R.reshape(M-1, N)
    part_Y = part_Y.reshape(M, N).T
    
    part_Y = part_Y @ W
    
    sol1 = np.zeros((M, N))

    sol = np.zeros(((2*M-1)*N, 1))
    
    for k in range(M):
        if k == 0: # Eigenvalue = 0, since sorted ascending
            
            # Find rhs
            f_loc = PETSc.Vec().create()
            f_loc.setSizes(N)
            f_loc.setType('mpi')
            f_loc.array[:] = part_Y[:, k]
            
            assert Lam_t[k] == 0
            
            # Solve for y
            y_loc = PETSc.Vec().create()
            y_loc.setSizes(N)
            y_loc.setType('mpi')
            
            solve_petsc(A_petsc, f_loc, y_loc, comm)
                        
            sol1[k, :] = y_loc.array[:]
            
            f_loc.destroy()
            y_loc.destroy()
        
        else:
            # Find rhs
            f_loc = PETSc.Vec().create()
            f_loc.setSizes(N)
            f_loc.setType('mpi')
            f_loc.array[:] = part_Y[:, k]
            
            lam_loc = Lam_t[k]
            
            # Solve for y
            y_loc = PETSc.Vec().create()
            y_loc.setSizes(N)
            y_loc.setType('mpi')
           
            A_petsc_edit = A_petsc.copy()
            A_petsc_edit.axpy(alpha=np.sqrt(lam_loc), X=M_petsc)
            
            solve_petsc(A_petsc_edit, f_loc, y_loc, comm)
            f_loc = A_petsc @ y_loc
            solve_petsc(A_petsc_edit, f_loc, y_loc, comm)
            
            sol1[k, :] = y_loc.array[:]
            
            
            f_loc.destroy()
            y_loc.destroy()
            A_petsc_edit.destroy()
            
    sol1 = W @ sol1
    new_rhs = Z_t@sol1@M_x + part_R
    sol[(M-1)*N:, 0] = sol1.reshape((M*N, 1))[:, 0]
    
    M_tD_diag = M_tD.diagonal()
    
    for k in range(M-1):
        # Find rhs
        f_loc = PETSc.Vec().create()
        f_loc.setSizes(N)
        f_loc.setType('mpi')
        f_loc.array[:] = new_rhs[k, :]
        
        m_loc = M_tD_diag[k]
        
        # Solve for y
        y_loc = PETSc.Vec().create()
        y_loc.setSizes(N)
        y_loc.setType('mpi')
        
        solve_petsc(A_petsc, f_loc, y_loc, comm)
        
        sol[(k)*N:(k+1)*N,0] = (1.0/m_loc)*y_loc.array[:]
        
        f_loc.destroy()
        y_loc.destroy()

    return sol




def assemble_mat_mu(mu, S_q, thetas_S):
    """
    Builds a parameter-dependent matrix based on a given parameter and an 
    offline-online-decomposition.

    Parameters
    ----------
    mu : numpy.ndarray
        The parameter for which the matrix is being built.
    S_q : list
        A list of parameter-independent matrix parts.
    thetas_S : list or function
        A list of functions or a function that returns a list of coefficients 
        to assemble the matrix with.

    Returns
    -------
    res : numpy.ndarray or scipy.sparse._csr.csr_matrix
        The assembled matrix.
    """
    
    if issparse(S_q[0]):
        res = csr_matrix((S_q[0].shape[0], S_q[0].shape[0]))
    else:
        res = np.zeros((S_q[0].shape[0], S_q[0].shape[0]))
    
    if isinstance(thetas_S, list):
        thetas_loc = []
        for theta in thetas_S:
            thetas_loc.append(theta(mu))
    else:
        thetas_loc = thetas_S(mu)
        
    assert len(thetas_loc) == len(S_q)
    for k in range(0, len(S_q)):
        res += thetas_loc[k]*S_q[k]
    
    if issparse(S_q[0]):
        res = res.tocsr()
    return res




def assemble_vec_mu(mu, s_q, thetas_s):
    """
    Builds a parameter-dependent vector based on a given parameter and an 
    offline-online-decomposition.

    Parameters
    ----------
    mu : numpy.ndarray
        The parameter for which the vector is being built.
    s_q : list
        A list of parameter-independent vector parts.
    thetas_s : function
        A function that returns a list of parameter-dependent coefficients.

    Returns
    -------
    res : numpy.ndarray
        The assembled vector.
    """
    
    res = np.zeros((s_q[0].shape[0],1))
    thetas_loc = thetas_s(mu)
    assert len(thetas_loc) == len(s_q)
    for k in range(0, len(s_q)):
        res += thetas_loc[k]*s_q[k]
    return res




prec_prepared = 0
def hf_solve_exe(mu, thetas_S, S_q, thetas_s, s_q, use_reformulation=False, 
                 use_preconditioner=True, comm=False, A_t=None, M_t=None, 
                 Z_t=None, M_tD=None, M_x=None, thetas_A=False, A_x_q=False, 
                 M=0, N=0):
    """
    This method is used to construct a method for solving the high fidelity 
    problem. The saddle point problem can be solved with or without 
    preconditioner, or it can be reduced to an equation for y only 
    ("reformulation"). We use GMRES for solving the saddle point problem with 
    preconditioner.

    Parameters
    ----------
    mu : numpy.ndarray
        The parameter for which the problem should be solved.
    thetas_S : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent system matrix.
    S_q : list
        The parameter-independent parts for the system matrix.
    thetas_s : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent right-hand side vector.
    s_q : list
        The parameter-independent parts for the right-hand side vector.
    use_reformulation : bool, optional
        Needs to be set to True, if the saddle point problem is reduced to an 
        equation for y only. In this case, S_q, thetas_S, s_q and thetas_s are 
        different. The default is False.
    use_preconditioner : bool, optional
        Decides, whether the preconditioner should be used. The default is 
        True.
    comm : mpi4py.MPI.Intracomm
        The Intracomm for PETSc. This is only needed if the preconditioner is 
        used for the saddle point problem. The default is False.
    A_t : scipy.sparse._dia.dia_matrix, optional
        The stiffness matrix in time. This is only needed if the preconditioner 
        is used for the saddle point problem. The default is None.
    M_t : scipy.sparse._dia.dia_matrix, optional
        The mass matrix in time. This is only needed if the preconditioner is 
        used for the saddle point problem. The default is None.
    Z_t : scipy.sparse._dia.dia_matrix, optional
        The CG1-DG0 transport matrix in time. This is only needed if the 
        preconditioner is used for the saddle point problem. The default is 
        None.
    M_tD : scipy.sparse._dia.dia_matrix, optional
        The DG0 mass matrix in time. This is only needed if the preconditioner 
        is used for the saddle point problem. The default is None.
    M_x : scipy.sparse._dia.dia_matrix or scipy.sparse._csr.csr_array, optional
        The mass matrix in space. This is only needed if the preconditioner is 
        used for the saddle point problem. The default is None.
    thetas_A : list, optional
        A list of parameter functions to assemble A(\mu). This is only needed 
        if the preconditioner is used for the saddle point problem. The default 
        is False.
    A_x_q : list, optional
        A list of matrices that represent A_q. This is only needed if the 
        preconditioner is used for the saddle point problem. The default is 
        False.
    M : int, optional
        Number of DOFs in time. This is only needed if no preconditioner is 
        used for the saddle point problem. The default is 0.
    N : int, optional
        Number of DOFs in space. This is only needed if no preconditioner is 
        used for the saddle point problem. The default is 0.

    Returns
    -------
    y : numpy.ndarray
        The DOF vector of the solution.
    """
    
    assert (not use_preconditioner or comm) or use_reformulation, \
        "Please provide an Intracomm."
    assert (not use_preconditioner or A_t is not None) or use_reformulation,\
        "Please provide the stiffness matrix in time."
    assert (not use_preconditioner or M_t is not None) or use_reformulation,\
        "Please provide the mass matrix in time."
    assert (not use_preconditioner or Z_t is not None) or use_reformulation,\
        "Please provide the transport matrix in time."
    assert (not use_preconditioner or M_tD is not None) or use_reformulation,\
        "Please provide the mass matrix in time with DG0 elements."
    assert (not use_preconditioner or M_x is not None) or use_reformulation,\
        "Please provide the mass matrix in space."
    assert (not use_preconditioner or thetas_A) or use_reformulation,\
        "Please provide the parameter functions for A."
    assert (not use_preconditioner or A_x_q) or use_reformulation,\
        "Please provide the matrices for A_q."
    assert use_reformulation or \
        (not use_preconditioner and (M != 0 and N != 0)) or\
            use_preconditioner, "Please provide the number of DOFs."
    
    S = assemble_mat_mu(mu, S_q, thetas_S)
    s = assemble_vec_mu(mu, s_q, thetas_s)
    if use_reformulation:
        return np.array([spsolve(S, s)]).T
    else:
        if use_preconditioner:
            global W, Lam_t, prec_prepared
            
            if prec_prepared != 0:
                if prec_prepared != A_t.shape[0]:
                    prec_prepared = 0
            if prec_prepared == 0:
                Lam_t, W = eigh(A_t.toarray(), M_t.toarray())
                Lam_t[0] = 0
                prec_prepared = A_t.shape[0]
                
            A_x_q_mu = []
            for i in range(0, len(thetas_A)):
                A_x_q_mu.append(thetas_A[i](mu) * A_x_q[i])
            A_x_mu = sum(A_x_q_mu).tocsc()
            
            A_petsc = scipy_to_petsc(A_x_mu)
            M_petsc = scipy_to_petsc(M_x)
            
            M = M_t.shape[0]
            N = M_x.shape[0]
            
            def solve_preconditioner(b):
                return preconditioner(b, N, M, W, Lam_t, A_petsc, M_petsc, 
                                      M_x, Z_t, M_tD, comm)

            prec = LinearOperator(((2*M-1)*N, (2*M-1)*N), solve_preconditioner)
        
            sol = gmres(S, s, M=prec, rtol=1e-5, atol=0.0, maxiter=10000)
            
            assert sol[1] == 0, "High fidelity solve not possible with GMRES. \
                Check formulation."

            y = sol[0][(M-1)*N:]
            return np.array([y]).T

        else:
            return np.array([spsolve(S, s)[(M-1)*N:]]).T




def prepare_correlation(y, M_x, A_x, Z_t, M_tD_inv):
    """
    Computes the discrete Riesz representer for the later use in the 
    correlation matrix in POD.

    Parameters
    ----------
    y : numpy.ndarray
        Space-time DOF vector.
    M_x : scipy.sparse._dia.dia_matrix or scipy.sparse._csr.csr_array
        The mass matrix in space.
    A_x : scipy.sparse._csr.csr_array
        The matrix representation of A for the reference parameter.
    Z_t : scipy.sparse._dia.dia_matrix
        The CG1-DG0 transport matrix in time.
    M_tD_inv : scipy.sparse._dia.dia_matrix
        The inverse of M_tD (which is diagonal).

    Returns
    -------
    Ry_t : numpy.ndarray
        The DOF vector of the Riesz representer of y_t.
    """
    
    M = Z_t.shape[1]
    N = int(y.size/M)
    y_mat = y.reshape((M, N)).T
    y_mat = M_x @ y_mat
    res = spsolve(A_x, y_mat)
    res = res @ Z_t.T
    res = res @ M_tD_inv    
    res = res.T
    return res.reshape((N*(M-1),1))




def build_W_inner_prod(A_x,M_x,A_t,M_t,T_t): 
    """
    Builds the true matrix representation of the W_d inner product. Very 
    costly, only used for a proof of concept.

    Parameters
    ----------
    A_x : scipy.sparse._csr.csr_array
        The matrix representation of A for the reference parameter.
    M_x : scipy.sparse._dia.dia_matrix or scipy.sparse._csr.csr_array
        The mass matrix in space.
    A_t : scipy.sparse._dia.dia_matrix
        The stiffness matrix in time for CG1 elements.
    M_t : scipy.sparse._dia.dia_matrix
        The mass matrix in time for CG1 elements.
    T_t : scipy.sparse._coo.coo_array
        The matrix \chi_j(T)\chi_i(T) for CG1 elements \chi.

    Returns
    -------
    scipy.sparse._bsr.bsr_matrix
        The matrix representation of the W_d inner product.
    """

    print("Building the W_d inner product.")
    return kron(A_t, M_x @ spsolve(A_x, M_x.tocsc())) \
        + kron(M_t, A_x) + kron(T_t, M_x)