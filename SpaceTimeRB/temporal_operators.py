#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains methods for assembling system matrices and vectors with 
respect to time. For research purposes and as we are interested in a proof of 
concept, we assemble the matrices and vectors manually for CG1-DG0 elements.
For practical reasons, we also recommend doing this with DOLFINx. The available
methods are:\n
- CG1_1D (basis function)
- DG0_1D (basis function)
- gauss_quadrature
- create_mass_CG1
- create_stiffness_CG1
- create_mass_DG0
- create_transport_CG1_DG0
- build_temporal_matrices_CG1DG0
- build_temporal_vectors_CG1DG0
"""

import numpy as np
from scipy.sparse import spdiags, coo_array




def CG1_1D(x, i, x_arr, evalGrad=False):
    """
    The one-dimensional CG1 basis function.

    Parameters
    ----------
    x : float
        Position at which to evaluate the basis function.
    i : int
        Index of the basis function to evaluate.
    x_arr : numpy.ndarray
        Grid.
    evalGrad : bool, optional
        Decides, whether the gradient of the basis function should be 
        evaluated. The default is False.

    Returns
    -------
    float
        Value of the basis function at x.
    """
    
    x_i = x_arr[i]
    
    if (i == 0):
        x_im1 = -np.Infinity
        x_ip1 = x_arr[i+1]
    elif (i == len(x_arr)-1):
        x_im1 = x_arr[i-1]
        x_ip1 = np.Infinity
    else:
        x_im1 = x_arr[i-1]
        x_ip1 = x_arr[i+1]
    
    if (not evalGrad):
        if (x < x_im1 or x >= x_ip1):
            return 0.0
        elif (x >= x_im1 and (((x <= x_i) and i != 0) 
                              or (x < x_i and i == 0))):
            return (x - x_im1)/(x_i - x_im1)
        else:
            return (x_ip1 - x)/(x_ip1 - x_i)
    else:
        if (x < x_im1 or x >= x_ip1):
            return 0.0
        elif (x >= x_im1 and (((x <= x_i) and i != 0) 
                              or (x < x_i and i == 0))):
            return 1.0/(x_i - x_im1)
        else:
            return -1.0/(x_ip1 - x_i)




def DG0_1D(x, i, x_arr):
    """
    The one-dimensional DG0 basis function.

    Parameters
    ----------
    x : float
        Position at which to evaluate the basis function.
    i : int
        Index of the basis function to evaluate.
    x_arr : numpy.ndarray
        Grid.

    Returns
    -------
    float
        Value of the basis function at x.
    """
    
    x_i = x_arr[i]
    
    if (i == len(x_arr)-1):
        x_ip1 = np.Infinity
    else:
        x_ip1 = x_arr[i+1]
    
    return x >= x_i and x <= x_ip1




def gauss_quadrature(f, dom_arr):
    """
    A method to integrate f over the discretized domain dom_arr.

    Parameters
    ----------
    f : function
        A function to integrate.
    dom_arr : numpy.ndarray
        A grid of the integration domain. The quadrature rule is applied on 
        each cell.

    Returns
    -------
    float
        Gauss-Legendre quadrature order 4.
    """
    
    points = [-np.sqrt(3.0/7.0 + (2.0/7.0)*np.sqrt(6.0/5.0)),
              -np.sqrt(3.0/7.0 - (2.0/7.0)*np.sqrt(6.0/5.0)),
              np.sqrt(3.0/7.0 - (2.0/7.0)*np.sqrt(6.0/5.0)),
              np.sqrt(3.0/7.0 + (2.0/7.0)*np.sqrt(6.0/5.0))]
    
    weights = [(18.0-np.sqrt(30))/36.0,
               (18.0+np.sqrt(30))/36.0,
               (18.0+np.sqrt(30))/36.0,
               (18.0-np.sqrt(30))/36.0]
    
    ret = 0
    for i in range(0, len(dom_arr)-1):
        x_i = dom_arr[i]
        x_ip1 = dom_arr[i+1]
        for j in range(0, len(points)):
            ret += ((x_ip1-x_i)/2.0) * weights[j] * \
                f(((x_ip1-x_i)/2.0)*points[j] + ((x_ip1+x_i)/2.0))
    return ret




def create_mass_CG1(ref_array):
    """
    Creates the CG1 mass matrix on a given one-dimensional grid.

    Parameters
    ----------
    ref_array : numpy.ndarray
        Grid.

    Returns
    -------
    M : scipy.sparse._dia.dia_matrix
        The mass matrix for CG1 elements.
    """
    
    N = ref_array.size-1
    data = np.zeros((3,N+1))
    for i in range(0, N):
        d = ref_array[i+1] - ref_array[i]
        data[0][i] += d/3.0
        if (i != N):
            data[0][i+1] += d/3.0
            data[1][i] = d/6.0
            data[2][i+1] = d/6.0
    return spdiags(data, [0, -1, 1])




def create_stiffness_CG1(ref_array):
    """
    Creates the CG1 stiffness matrix on a given one-dimensional grid.

    Parameters
    ----------
    ref_array : numpy.ndarray
        Grid.

    Returns
    -------
    A : scipy.sparse._dia.dia_matrix
        The stiffness matrix for CG1 elements.
    """
    
    N = ref_array.size-1
    data = np.zeros((3,N+1))
    for i in range(0, N):
        d_inv = 1.0/(ref_array[i+1] - ref_array[i])
        data[0][i] += d_inv
        if (i != N):
            data[0][i+1] += d_inv
            data[1][i] = -d_inv
            data[2][i+1] = -d_inv
    return spdiags(data, [0, -1, 1])




def create_mass_DG0(ref_array):
    """
    Creates the DG0 mass matrix on a given one-dimensional grid.

    Parameters
    ----------
    ref_array : numpy.ndarray
        Grid.

    Returns
    -------
    M : scipy.sparse._dia.dia_matrix
        The mass matrix for DG0 elements.
    """
    
    N = ref_array.size-1
    data = np.zeros((1,N))
    for i in range(0, N):
        data[0][i] = ref_array[i+1] - ref_array[i]
    return spdiags(data, [0])




def create_transport_CG1_DG0(ref_array):
    """
    Creates the transport matrix for CG1-DG0 elements.

    Parameters
    ----------
    ref_array : numpy.ndarray
        Grid.

    Returns
    -------
    Z : scipy.sparse._dia.dia_matrix
        The transport matrix for CG1-DG0 elements. It holds that (Z)_{ij} = 
        \int (\chi_j)_t \psi_i for CG1 elements \chi and DG0 elements \psi.
    """
    
    N = ref_array.size-1
    data = np.ones((2,N+1))
    data[0, :] *= -1
    return spdiags(data, [0, 1], m=N, n=N+1)




def build_temporal_matrices_CG1DG0(t_arr):
    """
    This method generates all the necessary matrices for a given time grid.

    Parameters
    ----------
    t_arr : numpy.ndarray
        A time grid.

    Returns
    -------
    A_t : scipy.sparse._dia.dia_matrix
        The stiffness matrix in time for CG1 elements.
    M_t : scipy.sparse._dia.dia_matrix
        The mass matrix in time for CG1 elements.
    T_t : scipy.sparse._coo.coo_array
        The matrix \chi_j(T)\chi_i(T) for CG1 elements \chi.
    Z_t : scipy.sparse._dia.dia_matrix
        The CG1-DG0 transport matrix.
    M_tD : scipy.sparse._dia.dia_matrix
        The mass matrix in time for DG0 elements.
    M_tD_inv : scipy.sparse._dia.dia_matrix
        The inverse of M_tD (which is diagonal).
    """
    
    M_t = create_mass_CG1(t_arr)
    n_t = t_arr.size-1
    T_t = coo_array((np.array([1.0]), (np.array([n_t]), np.array([n_t]))), 
                    shape=(n_t+1, n_t+1))
    A_t = create_stiffness_CG1(t_arr)
    M_tD = create_mass_DG0(t_arr)
    Z_t = create_transport_CG1_DG0(t_arr)
    M_tD_inv = spdiags([1.0/M_tD.diagonal()], [0])
    return A_t, M_t, T_t, Z_t, M_tD, M_tD_inv




def build_temporal_vectors_CG1DG0(t_arr, operators_f_t):
    """
    This method generates all the necessary vectors in time. We use the 
    Gauss-Legendre quadrature order 4 for integration.

    Parameters
    ----------
    t_arr : numpy.ndarray
        A time grid.
    operators_f_t : list
        A list of functions representing the temporal part f^t of f = f^tf^x.

    Returns
    -------
    R_0_t : numpy.ndarray
        A vector of the form \chi_j(0) for CG1 elements \chi.
    F_1_t_q : list
        A list of numpy.ndarray vectors of the form \int f^t \chi_j, where f^t 
        is the temporal part of f from L^2(0,T) and \chi_j is a CG1 basis 
        function.
    F_2_t_q : list
        A list of numpy.ndarray vectors of the form \int f^t \psi_j, where f^t 
        is the temporal part of f from L^2(0,T) and \psi_j is a DG0 basis 
        function.
    F_2_t_q_reformulation : list
        A list of numpy.ndarray vectors of the form \int f^t (\chi_j)_t, where 
        f^t is the temporal part of f from L^2(0,T) and \chi_j is a CG1 basis 
        function. This vector is used if the saddle point problem is reduced to
        an equation for y only.
    """
    
    n_t = t_arr.size-1
    R_0_t = np.zeros((n_t+1,1))
    R_0_t[0][0] = 1.0
    
    F_1_t_q = []
    for f_t in operators_f_t:
        F_1_t = np.zeros((n_t+1,1))
        for i in range(0, n_t+1):
            F_1_t[i][0] = gauss_quadrature(
                lambda t: f_t(t) * CG1_1D(t, i, t_arr), t_arr)
        F_1_t_q.append(F_1_t)
        
    F_2_t_q = []
    F_2_t_q_reformulation = []

    for f_t in operators_f_t:
        F_2_t = np.zeros((n_t,1))
        F_2_t_reformulation = np.zeros((n_t+1,1))
        for i in range(0, n_t+1):
            F_2_t_reformulation[i][0] = gauss_quadrature(
                lambda t: f_t(t) * CG1_1D(t, i, t_arr, evalGrad=True), t_arr)
            if i != n_t:
                F_2_t[i][0] = gauss_quadrature(
                    lambda t: f_t(t) * DG0_1D(t, i, t_arr), t_arr)
        F_2_t_q.append(F_2_t)
        F_2_t_q_reformulation.append(F_2_t_reformulation)
    
    return R_0_t, F_1_t_q, F_2_t_q, F_2_t_q_reformulation