#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a module containing all the methods needed to run the POD-greedy 
algorithm, as well as additional proof-of-concept methods. The available 
methods are:\n
- rb_solve
- min_theta
- max_theta
- estimator_part1
- estimator_part2
- prepare_fine_estimator
- fine_estimator
- POD_greedy
"""

import numpy as np
from numpy import linalg as la
from scipy.sparse import kron
from scipy.sparse.linalg import spsolve
from .high_fidelity import assemble_mat_mu, assemble_vec_mu, \
    prepare_correlation
from .offline_phase import POD, offline_phase




def rb_solve(mu, S_q_rb, thetas_S, s_q_rb, thetas_s, use_reformulation=True):
    """
    Solves the reduced problem.

    Parameters
    ----------
    mu : numpy.ndarray
        A parameter to solve the reduced problem for.
    S_q_rb : list
        The reduced system matrices.
    thetas_S : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent system matrix S_rb(\mu).
    s_q_rb : list
        The reduced right-hand side vectors.
    thetas_s : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent vector s_rb(\mu).
    use_reformulation : bool, optional
        If the saddle point equation has been reduced to an equation for y 
        only, this needs to be set to True. The default is True.

    Returns
    -------
    u_rb : numpy.ndarray
        The coefficient vector for the reduced solution represented in the 
        reduced basis.
    """
    
    S = assemble_mat_mu(mu, S_q_rb, thetas_S)
    s = assemble_vec_mu(mu, s_q_rb, thetas_s)
    res = la.solve(S, s)
    if use_reformulation:
        return res
    else:
        return res[int(res.size/2):]




def min_theta(mu, thetas_A, thetas_A_mu_bar):
    """
    A method for performing the min-theta approach to estimate the coercivity 
    constant of A.

    Parameters
    ----------
    mu : numpy.ndarray
        The current parameter.
    thetas_A : list
        A list of parameter functions which provide the coefficients required 
        to assemble A(\mu).
    thetas_A_mu_bar : list
        List of evaluated parameter functions theta_A^q for the reference 
        parameter.

    Returns
    -------
    float
        Results of the min-theta approach.
    """
    
    ret = np.infty
    for i in range(0, len(thetas_A)):
        ret = np.min([ret, thetas_A[i](mu)/thetas_A_mu_bar[i]])
        # Using that the coercivity constant of A(\mu_{ref}) is 1 per def.
    return ret




def max_theta(mu, thetas_A, thetas_A_mu_bar):
    """
    A method for performing the max-theta approach to estimate the continuity 
    constant of A. This works for specific examples only, e.g. thermal block 
    type examples.

    Parameters
    ----------
    mu : numpy.ndarray
        The current parameter.
    thetas_A : list
        A list of parameter functions which provide the coefficients required 
        to assemble A(\mu).
    thetas_A_mu_bar : list
        List of evaluated parameter functions theta_A^q for the reference 
        parameter.

    Returns
    -------
    float
        Results of the max-theta approach.
    """
    
    ret = -np.infty
    for i in range(0, len(thetas_A)):
        ret = np.max([ret, thetas_A[i](mu)/thetas_A_mu_bar[i]])
        # Using that the continuity constant of A(\mu_{ref}) is 1 per def.
    return ret




def estimator_part1(u_rb, reduced_basis, G, mu=None, 
                    thetas_S_reformulation=None, thetas_s_reformulation=None, 
                    thetas_S_loc=None, thetas_s_loc=None):
    """
    This is a method that returns the numerator part \sqrt(r.T @ G @ r) of the 
    error estimator.

    Parameters
    ----------
    u_rb : numpy.ndarray
        The coefficient vector for the reduced solution represented in the 
        reduced basis.
    reduced_basis : list
        List of DOF vectors of the reduced basis for y.
    G : numpy.ndarray
        Matrix for computing the residual estimation in the reduced basis 
        space.
    mu : numpy.ndarray, optional
        The current parameter. This only needs to be provided if the evaluated 
        thetas_S_loc and thetas_s_loc for the reformulation are not given. The 
        default is None.
    thetas_S_reformulation : function, optional
        Function that returns a list of parameter-dependent coefficients for 
        the matrices S_q_reformulation. This only needs to be provided if the 
        evaluated thetas_S_loc and thetas_s_loc for the reformulation are not 
        given. The default is None.
    thetas_s_reformulation : function, optional
        Function that returns a list of parameter-dependent coefficients for 
        the vectors s_q_reformulation. This only needs to be provided if the 
        evaluated thetas_S_loc and thetas_s_loc for the reformulation are not 
        given. The default is None.
    thetas_S_loc : list, optional
        A list of parameter-dependent coefficients for assembling the 
        parameter-dependent system matrix S(\mu) in the reformulation. The 
        default is None.
    thetas_s_loc : list, optional
        A list of parameter-dependent coefficients for assembling the 
        parameter-dependent vector s(\mu) in the reformulation. The default is 
        None.

    Returns
    -------
    float
        The first part of the estimator.
    """
    
    assert (thetas_S_loc is not None and thetas_s_loc is not None) or \
        (thetas_S_reformulation is not None and \
         thetas_s_reformulation is not None and mu is not None)
    
    if thetas_S_loc is None:
        thetas_S_loc = thetas_S_reformulation(mu)
    if thetas_s_loc is None:
        thetas_s_loc = thetas_s_reformulation(mu)
        
    r = np.zeros((1, len(thetas_s_loc)+len(thetas_S_loc)*len(reduced_basis)))
    
    r[0, :len(thetas_s_loc)] = thetas_s_loc[:]        
    index = len(thetas_s_loc)
    for i in range(0, len(thetas_S_loc)):
        r[0, index:index+len(reduced_basis)] = (-1.0)*thetas_S_loc[i]*u_rb.T[0] 
        index += len(reduced_basis)
    r = r.T
    
    val = (r.T @ G @ r)[0][0]
    if val < 0:
        val = 0 # Remove numerical error +/- 1e-17 that produces np.nan.
    return np.sqrt(val)




def estimator_part2(mu, thetas_A, thetas_A_mu_bar, max_theta_applicable=True, 
                    c_s=0):
    """
    This is a method that returns the denominator part of the error estimator.

    Parameters
    ----------
    mu : numpy.ndarray
        The current parameter.
    thetas_A : list
        A list of parameter functions which provide the coefficients required 
        to assemble A(\mu).
    thetas_A_mu_bar : list
        List of evaluated parameter functions theta_A^q for the reference 
        parameter.
    max_theta_applicable : bool, optional
        Decides, whether a max-theta approach can be used to estimate the 
        continuity constant. The default is True.
    c_s : float, optional
        The parameter-independent continuity constant. This only needs to be 
        provided if the max-theta approach is not applicable. The default is 0.

    Returns
    -------
    float
        The second part of the error estimator.
    """
    
    assert c_s != 0 or max_theta_applicable, \
        "Please check the problem formulation."
    
    min_theta_loc = min_theta(mu, thetas_A, thetas_A_mu_bar)
    if c_s != 0:
        max_theta_loc = c_s
    if max_theta_applicable:
        max_theta_loc = max_theta(mu, thetas_A, thetas_A_mu_bar)

    return 1.0 / (np.min([min_theta_loc, 1.0/max_theta_loc]) * min_theta_loc)




def prepare_fine_estimator(A_x, A_x_q):
    """
    Computes A_x^{-1}A_x_q for the use in the fine estimator. This is a method 
    for proof of concept only.

    Parameters
    ----------
    A_x : scipy.sparse._csr.csr_array
        The matrix representation of A for the reference parameter.
    A_x_q : list
        A list of matrices representing A_q.

    Returns
    -------
    A_x_q_solved : list
        A list of matrices A_x^{-1}A_x_q.
    """
    
    A_x_q_solved = []
    for A_q in A_x_q:
        A_x_q_solved.append(spsolve(A_x, A_q))
    return A_x_q_solved




def fine_estimator(mu, u_rb, reduced_basis, K_rb, A_x, A_x_q, A_x_q_solved, 
                   thetas_A, thetas_A_mu_bar, M_x_inv, A_t, M_t, T_t,
                   S_q_reformulation, thetas_S_reformulation, 
                   s_q_reformulation, thetas_s_reformulation,
                   rel_estimator=False, max_theta_applicable=True, c_s=0):
    """
    Evaluates the fine error estimator with true residual norm. This method is 
    for proof of concept only.
    
    Parameters
    ----------
    mu : numpy.ndarray
        The current parameter.
    u_rb : numpy.ndarray
        The coefficient vector for the reduced solution represented in the 
        reduced basis.
    reduced_basis : list
        List of DOF vectors of the reduced basis for y.
    K_rb : numpy.ndarray
        Matrix representation of the inner product in the reduced basis space.
    A_x : scipy.sparse._csr.csr_array
        The matrix representation of A for the reference parameter.
    A_x_q : list
        A list of matrices representing A_q.
    A_x_q_solved : list
        A list of matrices A_x^{-1}A_x_q.
    thetas_A : list
        A list of functions that return theta_A^q(mu) for a given parameter mu.
    thetas_A_mu_bar : list
        List of evaluated parameter functions theta_A^q for the reference 
        parameter.
    M_x_inv : scipy.sparse._dia.dia_matrix
        Inverted spatial mass matrix.
    A_t : scipy.sparse._dia.dia_matrix
        The stiffness matrix in time for CG1 elements.
    M_t : scipy.sparse._dia.dia_matrix
        The mass matrix in time for CG1 elements.
    T_t : scipy.sparse._coo.coo_array
        The matrix \chi_j(T)\chi_i(T) for CG1 elements \chi.
    S_q_reformulation : list
        List of parameter-independent space-time system matrices S_q for the 
        reformulation, where the saddle point problem is reduced to an equation 
        for y only.
    thetas_S_reformulation : function
        Function that returns a list of parameter-dependent coefficients for 
        the matrices S_q_reformulation.
    s_q_reformulation : list
        List of parameter-independent space-time right-hand side vectors s_q of 
        the reformulation.
    thetas_s_reformulation : function
        Function that returns a list of parameter-dependent coefficients for 
        the vectors s_q_reformulation.
    rel_estimator : bool, optional
        Set to True, if the relative error estimator should be used. The 
        default is False.
    max_theta_applicable : bool, optional
        Decides, whether a max-theta approach can be used to estimate the 
        continuity constant. The default is True.
    c_s : float, optional
        The parameter-independent continuity constant. This only needs to be 
        provided if the max-theta approach is not applicable. The default is 0.

    Returns
    -------
    eta : float
        The value of the error estimator.
    """
    
    assert c_s != 0 or max_theta_applicable, \
        "Please check the problem formulation."
    
    B = np.array(reduced_basis).T[0]
      
    A_x_mu = assemble_mat_mu(mu, A_x_q, thetas_A)
    A_x_mu_solved = assemble_mat_mu(mu, A_x_q_solved, thetas_A)
    system_matrix = kron(A_t, A_x_mu@A_x_mu_solved) \
        + kron(M_t, A_x_mu@M_x_inv@A_x@M_x_inv@A_x_mu) \
            +kron(T_t, A_x_mu@M_x_inv@A_x_mu)
            
    S_mu = assemble_mat_mu(mu, S_q_reformulation, thetas_S_reformulation)
    s_mu = assemble_vec_mu(mu, s_q_reformulation, thetas_s_reformulation)
    
    y_rb = B @ u_rb
    r = s_mu - S_mu@y_rb
    res = spsolve(system_matrix, r)
    res_norm = np.dot(res, r.T[0])
    res_norm = np.sqrt(res_norm)
    
    if rel_estimator:
        res_norm = 2.0 * res_norm / np.sqrt(u_rb.T @ K_rb @ u_rb)[0,0]
        
    c_c_mu = min_theta(mu, thetas_A, thetas_A_mu_bar)
    if c_s != 0:
        c_s_mu = c_s
    if max_theta_applicable:
        c_s_mu = max_theta(mu, thetas_A, thetas_A_mu_bar)

    alpha_mu = min(c_c_mu, 1/c_s_mu)

    return res_norm/alpha_mu




def POD_greedy(mu_start, S_train, reduced_basis, Z, K_rb, G, thetas_A, 
               thetas_A_mu_bar, thetas_S_reformulation, S_q_reformulation, 
               thetas_s_reformulation, s_q_reformulation, S_q, S_q_rb, 
               thetas_S_rb, s_q, s_q_rb, thetas_s_rb, hf_solve, M_x, M_x_inv, 
               A_x, A_x_q, A_t, M_t, T_t, Z_t, M_tD_inv, inner_prod_part1, 
               inner_prod_part2, inner_prod_res, use_reformulation=True, 
               conv_tol=1e-10, max_L=15, L1=1, L2=2, use_rel_estimator=False, 
               eval_true_err=True, eval_fine_est=True, A_x_q_solved = None,
               mu_val_set=None, y_val_set=None, max_theta_applicable=True, 
               c_s=0, inner_prod_ref=None):
    """
    The POD-greedy algorithm for the generation of reduced basis spaces.

    Parameters
    ----------
    mu_start : numpy.ndarray
        A start parameter for which a solution is available.
    S_train : list
        A training set of parameters.
    reduced_basis : list
        List of (high fidelity) DOF vectors of the reduced basis for y. The 
        basis should consist of the normed start solution.
    Z : list
        Collection of all computed high-fidelity solutions and their Riesz 
        representers in the form of a list of tuples. Only the solution for the 
        start parameter should be contained here.
    K_rb : numpy.ndarray
        Matrix representation of the inner product in the reduced basis space.
    G : numpy.ndarray
        Matrix for computing the residual estimation in the reduced basis 
        space.
    thetas_A : list
        A list of functions that return theta_A^q(mu) for a given parameter mu.
    thetas_A_mu_bar : list
        List of evaluated parameter functions theta_A^q for the reference 
        parameter.
    thetas_S_reformulation : function
        Function that returns a list of parameter-dependent coefficients for 
        the matrices S_q_reformulation.
    S_q_reformulation : list
        List of parameter-independent space-time system matrices S_q for the 
        reformulation, where the saddle point problem is reduced to an equation 
        for y only.
    thetas_s_reformulation : function
        Function that returns a list of parameter-dependent coefficients for 
        the vectors s_q_reformulation.
    s_q_reformulation : list
        List of parameter-independent space-time right-hand side vectors s_q of 
        the reformulation.
    S_q : list
        A list of high fidelity system matrices S_q.
    S_q_rb : list
        The reduced system matrices.
    thetas_S_rb : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent system matrix S(\mu). This should 
        correspond to the reduced system.
    s_q : list
        A list of high fidelity vectors s_q.
    s_q_rb : list
        The reduced right-hand side vectors.
    thetas_s_rb : function
        A function that returns a list of parameter-dependent coefficients for 
        assembling the parameter-dependent vector s(\mu). This should 
        correspond to the reduced system.
    hf_solve : function
        A local function that returns the solution for a given parameter. This 
        function should be based on hf_solve_exe.
    M_x : scipy.sparse._dia.dia_matrix
        Mass matrix with respect to space.
    M_x_inv : scipy.sparse._dia.dia_matrix
        Inverted mass matrix.
    A_x : scipy.sparse._csr.csr_array
        The matrix representation of A for the reference parameter.
    A_x_q : list
        A list of matrices representing A_q.
    A_t : scipy.sparse._dia.dia_matrix
        The stiffness matrix in time.
    M_t : scipy.sparse._dia.dia_matrix
        The mass matrix in time.
    T_t : scipy.sparse._coo.coo_array
        The matrix \chi_j(T)\chi_i(T) for CG1 elements \chi.
    Z_t : scipy.sparse._dia.dia_matrix
        The CG1-DG0 transport matrix in time.
    M_tD_inv : scipy.sparse._dia.dia_matrix
        The inverse of M_tD (which is diagonal).
    inner_prod_part1 : scipy.sparse._csr.csr_array
        First part of the W_d inner product.
    inner_prod_part2 : scipy.sparse._coo.coo_matrix
        Second part of the W_d inner product.
    inner_prod_res : scipy.sparse._csr.csr_array
        Matrix for computing the residual norm estimation.
    use_reformulation : bool, optional
        Needs to be set to True, if the saddle point problem has been reduced 
        to an equation for y only. The default is True.
    conv_tol : bool, optional
        The convergence tolerance for the POD-greedy algorithm. The default is 
        1e-10.
    max_L : int, optional
        The maximal reduced basis space dimension. The default is 15.
    L1 : int, optional
        The amount of basis functions that should be added in each iteration. 
        The default is 1.
    L2 : int, optional
        The amount of high fidelity system solves per iteration. The default is 
        2.
    use_rel_estimator : bool, optional
        Set to True, if the relative error estimator should be used. The 
        default is False.
    eval_true_err : bool, optional
        Set to True, if the true W_d error should be computed for proof of 
        concept. The default is True.
    eval_fine_est : bool, optional
        Set to True, if the estimator with fine residual should be evaluated 
        for proof of concept. The default is True.
    A_x_q_solved : list, optional
        A list of matrices A_x^{-1}A_x_q. This needs to be provided if the fine 
        estimator is to be evaluated. The default is None.
    mu_val_set : list, optional
        A validation set. The default is None.
    y_val_set : list, optional
        A list of high fidelity solutions corresponding to the validation set. 
        The default is None.
    max_theta_applicable : bool, optional
        Decides, whether a max-theta approach can be used to estimate the 
        continuity constant. The default is True.
    c_s : float, optional
        The parameter-independent continuity constant. This only needs to be 
        provided if the max-theta approach is not applicable. The default is 0.
    inner_prod_ref : scipy.sparse._bsr.bsr_matrix, optional
        The matrix representation of the W_d inner product. This is only 
        required if the true error is to be evaluated. The default is None.

    Returns
    -------
    reduced_basis : list
        List of (high fidelity) DOF vectors of the new reduced basis for y.
    new_rb_Riesz : list
        List of DOF vectors of the Riesz representer of the new reduced basis.
    K_rb : numpy.ndarray
        Matrix representation of the inner product in the new reduced basis 
        space.
    S_q_rb : list
        The reduced system matrices.
    s_q_rb : list
        The reduced right-hand side vectors.
    """
    
    assert not eval_fine_est or A_x_q_solved is not None, \
        "Please provide A_x_q_solved."
    assert not eval_true_err or inner_prod_ref is not None, \
        "Please provide the W_d inner product."
    assert mu_val_set is None or inner_prod_ref is not None, \
        "Please provide the W_d inner product."
    
    print("Start Algorithm")
    iteration_L = 1
    tol_bool = False # for convergence criterion
    
    current_mu = mu_start
    chosen_mus = [current_mu]
    
    err_estimator = []
    err_real = []
    err_estimator_fine = []

    val_set_errors = []
    val_set_rough_eff = []
    val_set_fine_eff = []
    
    while iteration_L < max_L and not tol_bool:
        
        print("POD-greedy start to obtain W_" + str(iteration_L+L1))
        candidates = []
        
        for i in range(0, len(S_train)):
            mu_candidate = S_train[i]
        
            if not(isinstance(mu_candidate, (list, tuple, np.ndarray))):
                continue
            
            u_loc = rb_solve(mu_candidate, S_q_rb, thetas_S_rb, s_q_rb, 
                             thetas_s_rb, use_reformulation=use_reformulation)
            
            estimator = estimator_part1(
                u_loc, reduced_basis, G, mu=mu_candidate, 
                thetas_S_reformulation=thetas_S_reformulation, 
                thetas_s_reformulation=thetas_s_reformulation)\
                *estimator_part2(mu_candidate, thetas_A, thetas_A_mu_bar, 
                                 max_theta_applicable=max_theta_applicable,
                                 c_s=c_s)
            
            if use_rel_estimator:
                estimator = 2.0*estimator/np.sqrt(u_loc.T @ K_rb @ u_loc)[0,0]
            
            candidates.append((estimator, mu_candidate, i))
        
        candidates.sort(key=lambda t: -t[0])
        
        for l2 in range(0,L2):
            new_mu = candidates[l2][1]
            chosen_mus.append(new_mu)
            S_train[candidates[l2][2]] = None
            
            print("Solving HF node " + str(l2+1))
            new_node_hf = hf_solve(new_mu)
            print("Solving Corr-Helper")
            new_node_prep = prepare_correlation(
                new_node_hf, M_x, A_x, Z_t, M_tD_inv)
            
            Z.append((new_node_hf, new_node_prep))
            
            if l2 == 0:
                err_estimator.append(candidates[l2][0])
                np.savetxt("err_estimator.csv", err_estimator, delimiter=",")
                tol_bool = candidates[l2][0] <= conv_tol
                
                if eval_true_err:
                    print("Computation done, calculating the true error")       
                    u_loc = rb_solve(
                        new_mu, S_q_rb, thetas_S_rb, s_q_rb, thetas_s_rb, 
                        use_reformulation=use_reformulation)
                    B = np.array(reduced_basis).T[0]
                    new_node_rb = B @ u_loc
                    err_rb = new_node_hf - new_node_rb
                    err_real_here = np.sqrt(
                        (err_rb.T @ inner_prod_ref @ err_rb)[0, 0])
                    if use_rel_estimator:
                        err_real_here = err_real_here / np.sqrt(
                            (new_node_hf.T@inner_prod_ref@new_node_hf)[0, 0])
                    err_real.append(err_real_here)
                    np.savetxt("err_real.csv", err_real, delimiter=",")
                    tol_bool = err_real_here <= conv_tol
                
                if eval_fine_est:
                    fine_est = fine_estimator(
                        new_mu, u_loc, reduced_basis, K_rb, A_x, A_x_q, 
                        A_x_q_solved, thetas_A, thetas_A_mu_bar, M_x_inv, A_t, 
                        M_t, T_t, S_q_reformulation, thetas_S_reformulation, 
                        s_q_reformulation, thetas_s_reformulation, 
                        rel_estimator=use_rel_estimator, 
                        max_theta_applicable=max_theta_applicable, c_s=c_s)

                    err_estimator_fine.append(fine_est)
                    np.savetxt("thm_estimator_real_res.csv", 
                               err_estimator_fine, delimiter=",")
                    tol_bool = tol_bool or fine_est <= conv_tol
                
                if mu_val_set is not None and y_val_set is not None:
                    print("Computing errors for the validation set")
                    B = np.array(reduced_basis).T[0]
                    err_val_set = 0
                    eff_total = 0
                    eff_fine_total = 0
                    
                    for k_val in range(0, len(mu_val_set)):
                        u_val_loc = rb_solve(
                            mu_val_set[k_val], S_q_rb, thetas_S_rb, s_q_rb, 
                            thetas_s_rb, use_reformulation=use_reformulation)
                        estimator_val_loc = estimator_part1(
                            u_val_loc, reduced_basis, G, mu=mu_val_set[k_val], 
                            thetas_S_reformulation=thetas_S_reformulation, 
                            thetas_s_reformulation=thetas_s_reformulation)\
                            *estimator_part2(
                                mu_val_set[k_val], thetas_A, thetas_A_mu_bar, 
                                max_theta_applicable=max_theta_applicable, 
                                c_s=c_s)
                        if use_rel_estimator:
                            estimator_val_loc = 2.0*estimator_val_loc/np.sqrt(
                                u_val_loc.T @ K_rb @ u_val_loc)[0,0]
                        
                        err_loc_val = y_val_set[k_val] - B @ u_val_loc
                        err_loc_norm = np.sqrt(
                            (err_loc_val.T@inner_prod_ref@err_loc_val)[0, 0])
                        if use_rel_estimator:
                            err_loc_norm /= np.sqrt((y_val_set[k_val].T \
                                @ inner_prod_ref @ y_val_set[k_val])[0, 0])
                        
                        err_val_set += err_loc_norm
                        eff_total += estimator_val_loc / err_loc_norm
                        
                        if eval_fine_est:
                            
                            fine_est_val_loc = fine_estimator(
                                mu_val_set[k_val], u_val_loc, reduced_basis, 
                                K_rb, A_x, A_x_q, A_x_q_solved, thetas_A, 
                                thetas_A_mu_bar, M_x_inv, A_t, M_t, T_t, 
                                S_q_reformulation, thetas_S_reformulation, 
                                s_q_reformulation, thetas_s_reformulation, 
                                rel_estimator=use_rel_estimator, 
                                max_theta_applicable=max_theta_applicable, 
                                c_s=c_s)

                            eff_fine_total += fine_est_val_loc/err_loc_norm
                        
                    err_val_set /= len(mu_val_set)
                    eff_total /= len(mu_val_set)
                    eff_fine_total /= len(mu_val_set)

                    val_set_errors.append(err_val_set)
                    val_set_rough_eff.append(eff_total)
                    val_set_fine_eff.append(eff_fine_total)
                    np.savetxt("val_set_errors.csv", 
                               val_set_errors, delimiter=",")    
                    np.savetxt("val_set_rough_eff.csv", 
                               val_set_rough_eff, delimiter=",")
                    if eval_fine_est:
                        np.savetxt("val_set_fine_eff.csv", 
                                   val_set_fine_eff, delimiter=",")
                        
        print("Performing POD")
        iteration_L += L1
        reduced_basis, K_rb, new_rb_Riesz = POD(
            Z, iteration_L, inner_prod_part1, inner_prod_part2)
        
        print("Running the offline phase")
        G, S_q_rb, s_q_rb = offline_phase(
            reduced_basis, S_q, s_q, inner_prod_res, 
            use_reformulation=use_reformulation, 
            S_q_reformulation=S_q_reformulation, 
            s_q_reformulation=s_q_reformulation, 
            reduced_basis_Riesz=new_rb_Riesz)
        
    return reduced_basis, new_rb_Riesz, K_rb, S_q_rb, s_q_rb