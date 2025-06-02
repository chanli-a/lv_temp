"""
This file contains the code used for MiCRM and ELVM simuations. 

"""





import numpy as np
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from numba import prange
from numpy.linalg import norm
import os
import pandas as pd
import multiprocessing as mp
from scipy.integrate import solve_ivp
import pickle




# generate semi-random TPCs for consumer uptake (u) and respiration (m) 

def randtemp_param(N, kw):
    rng=kw.get('rng',np.random) 

    L = kw['L'] 
    rho_t = kw['rho_t']
    L_v = np.mean(L)
    B0_m = -1.4954 
    B0_CUE = 0.1953
    B0_u = np.log(np.exp(B0_m) / (1 - L_v - B0_CUE)) 
    B0 = np.array([B0_u, B0_m]) 
    B0_var = 0.17 * np.abs(B0) 
    E_mean = np.array([0.8146, 0.5741]) 
    E_var = 0.1364 * E_mean 
    cov_xy = rho_t * np.sqrt(B0_var * E_var)

    cov_u = np.array([[B0_var[0], cov_xy[0]], [cov_xy[0], E_var[0]]]) 
    cov_m = np.array([[B0_var[1], cov_xy[1]], [cov_xy[1], E_var[1]]]) 

    allu = multivariate_normal.rvs(mean=[B0[0], E_mean[0]], cov=cov_u, size=N).T 
    allm = multivariate_normal.rvs(mean=[B0[1], E_mean[1]], cov=cov_m, size=N).T 

    B = np.column_stack((np.exp(allu[0]), np.exp(allm[0]))) 
    E = np.column_stack((allu[1], allm[1])) 

    Tpu = 273.15 + rng.normal(35, 5, N) 
    Tpm = Tpu + 3 
    Tp = np.column_stack((Tpu, Tpm)) 

    return B, E, Tp




def temp_trait(N, kw):
    
    k = 0.0000862 
    T = kw['T']
    Tr = kw['Tr']
    Ed = kw['Ed']

    B, E, Tp = randtemp_param(N, kw) 

    # uptake rate u(T)
    temp_p_u = B[:, 0] * np.exp((-E[:, 0] / k) * ((1 / T) - (1 / Tr))) / \
              (1 + (E[:, 0] / (Ed - E[:, 0])) * np.exp(Ed / k * (1 / Tp[:, 0] - 1 / T)))

    # respiration rate m(T)
    temp_p_m = B[:, 1] * np.exp((-E[:, 1] / k) * ((1 / T) - (1 / Tr))) / \
              (1 + (E[:, 1] / (Ed - E[:, 1])) * np.exp(Ed / k * (1 / Tp[:, 1] - 1 / T)))

    temp_p = np.column_stack((temp_p_u, temp_p_m))  

    return temp_p, B, E, Tp

    

# default functions

def def_m(N, M, kw):
    return np.ones(N)


def def_rho(N, M, kw):
    return np.ones(M)


def def_omega(N, M, kw):
    return np.ones(M)


def def_u(N, M, kw):
    rng = kw.get('rng', np.random)
    return rng.dirichlet(np.ones(M), size=N)



def def_l(N, M, kw):
    L = kw['L']
    rng = kw.get('rng', np.random) 
    l = np.zeros((N, M, M))
    phi = np.ones(M)
    for i in range(N):
        for alpha in range(M):
            draw = rng.dirichlet(alpha=phi)
            l[i, alpha, :] = draw * L[i]
    return l



# generate parameters 

def generate_params(N,
                     M,
                     f_m=def_m,
                     f_rho=def_rho,
                     f_omega=def_omega,
                     f_u=def_u,
                     f_l=def_l,
                     **kwargs):

  


    kw = dict(kwargs)
    tt, B, E, Tp = temp_trait(N, kw) 
    kw['tt'] = tt

 
    m = f_m(N, M, kw) 
    u = f_u(N, M, kw) 

 
    l = f_l(N, M, kw)     


    lambda_ = np.sum(l, axis=2) 

 
    rho = f_rho(N, M, kw)
    omega = f_omega(N, M, kw)


    params = {
        'N': N,
        'M': M,
        'u': u,
        'm': m,
        'l': l,
        'rho': rho,
        'omega': omega,
        'lambda': lambda_,
        'L': L,
        'B': B,
        'E': E,
        'Tp': Tp,
        'tt': tt
    }
   
    params.update(kwargs)

    return params 

    



# defining the MiCRM function 

def MiCRM_dxx(x, t, N, M, u, l, rho, omega, m):
   
    dx = np.zeros(N + M)
    # consumer 
    for i in range(N):
        dx[i] = -m[i] * x[i]
        for alpha in range(M):
            res_idx = N + alpha
            uptake = x[i] * x[res_idx] * u[i, alpha]
            dx[i] += uptake
            for beta in range(M):
                dx[i] -= uptake * l[i, alpha, beta]
    # resource 
    for alpha in range(M):
        idx = N + alpha
        dx[idx] = rho[alpha] - omega[alpha] * x[idx]
        for i in range(N):
            dx[idx] -= u[i, alpha] * x[idx] * x[i]
            for beta in range(M):
                dx[idx] += x[N + beta] * x[i] * u[i, beta] * l[i, beta, alpha]
    return dx


def MiCRM_dxx_numba_wrapper(t, x, p): # numba for efficiency 
    return MiCRM_dxx(x, t,
                         p['N'], p['M'],
                         p['u'], p['l'],
                         p['rho'], p['omega'],
                         p['m'])
    

# vectorised code for efficiency 

# parameters for ELVM 

def eff_LV_params(p, sol, verbose=False):
    M, N = p['M'], p['N']
    l    = p['l']
    rho  = p['rho']
    omega= p['omega']
    m    = p['m']
    u    = p['u']
    lam  = p['lambda']

    Ceq = sol.y[:N, -1]
    Req = sol.y[N:, -1]

    A = -np.diag(omega)
    W = u * Ceq[:, None]
    A += np.einsum('ib, iab -> ab', W, l)
    diag_sub = W.sum(axis=0)
    A[np.diag_indices(M)] -= diag_sub

    invA = np.linalg.inv(A)

    eyeM = np.eye(M)
    D = (eyeM[None,:,:] - l)
    T = u[:,:,None] * Req[None,None,:] * D
    S = T.sum(axis=2)
    dR = (invA @ S.T)

    A_thing = u * (1 - lam)
    alpha = A_thing @ dR

    O = A_thing @ Req
    P = alpha @ Ceq
    r_eff = O - P - m

    result = {'alpha': alpha, 'r': r_eff, 'N': N}
    if verbose:
        result.update({'dR': dR, 'A': A})
    return result


def LV_dx(x, t, p):
    r     = p['r']
    alpha = p['alpha']
    interaction = alpha.dot(x)
    dx = x * (r + interaction)
    return dx


# Jacobian matrices

def eff_LV_jac(p_lv, sol, threshold=1e-7):
    alpha_full = p_lv['alpha']
    N_full     = p_lv['N']
    bm         = sol.y[:N_full, -1]
    feasible = np.where(bm > threshold)[0]
    if feasible.size == 0:
        raise ValueError("No feasible species found!")
    C    = bm[feasible]
    alpha = alpha_full[np.ix_(feasible, feasible)]
    J = np.diag(C) @ alpha
    return J

def MiCRM_jac(p, sol):
    N, M = p['N'], p['M']
    lam   = p['lambda']
    l     = p['l']
    omega = p['omega']
    m     = p['m']
    u     = p['u']
    state = sol.y[:, -1]
    C     = state[:N]
    R     = state[N:]
    cc_diag = -m + ((1 - lam) * u * R[None, :]).sum(axis=1)
    CC = np.diag(cc_diag)
    CR = C[:, None] * (1 - lam) * u
    P = C[:, None, None] * u[:, :, None] * l
    RR = P.sum(axis=0)
    diag_val = np.diag(RR)
    sub_diag = (C[:, None] * u).sum(axis=0)
    diag_rr = diag_val - sub_diag - omega
    np.fill_diagonal(RR, diag_rr)
    Q = u * R[None, :]
    Ql = Q[:, :, None] * l
    term2 = Ql.sum(axis=1)
    RC = (term2 - Q).T
    top    = np.hstack([CC, CR])
    bottom = np.hstack([RC, RR])
    return np.vstack([top, bottom])


# calculating eigenvalues (can be used for both MiCRM and ELVM)
def leading_eigenvalue(J): # stability 
    eigvals = np.linalg.eigvals(J)
    return eigvals[np.argmax(np.real(eigvals))]

def hermitian_part(J): 
    return (J + J.T) / 2

def leading_hermitian_eigenvalue(J): # reactivity 
    H = hermitian_part(J)
    eigvals = np.linalg.eigvalsh(H)
    return np.max(eigvals)




def F_m(N, M, kw):
    
    if 'tt' in kw:      
        return kw['tt'][:, 1] 
    else:
        return np.full(N, 0.2)


def F_rho(N, M, kw):   
    return np.ones(M)


def F_omega(N, M, kw):    
    return np.zeros(M)


def F_u(N, M, kw):
    rng=kw.get('rng',np.random)

    diri = np.stack([rng.dirichlet(np.ones(M)) for _ in range(N)], axis=0)

    if 'tt' in kw:
        u_sum = kw['tt'][:, 0] 
    else:
        u_sum = np.full(N, 2.5)

    return diri * u_sum[:, None]



# diversity metrics

def shannon(abundance): 
    C_shannon = np.asarray(abundance, dtype=float)
    total_abundance = np.sum(C_shannon)
    pi = C_shannon / total_abundance
    pi_lnpi = pi[pi > 0] * np.log(pi[pi > 0])
    H = -np.sum(pi_lnpi)
    return H


def bray_curtis_dissimilarity(G, M): 
    G_array = np.asarray(G, dtype=float)
    M_array = np.asarray(M, dtype=float)
    G_safe = np.where(G_array < 0, 0, G_array)
    M_safe = np.where(M_array < 0, 0, M_array)
    GM_dissimilarity = np.sum(np.abs(G_safe - M_safe)) / np.sum(G_safe + M_safe)
    return GM_dissimilarity


def jaccard_index(G, M, thresh=1e-8):
    G = np.asarray(G, dtype=float)
    M = np.asarray(M, dtype=float)
    if G.shape != M.shape:
        raise ValueError("G and M should have the same shape")
    G_surv = G > thresh
    M_surv = M > thresh
    inter = np.logical_and(G_surv, M_surv).sum()
    union = np.logical_or(G_surv, M_surv).sum()
    return 1.0 if union == 0 else inter / union


# equilibrium abundance deviations (only surviving species)

def err_eq_and_overlap(C_LV_eq, C_MiCRM_eq, thresh=1e-6):
    C_LV = np.asarray(C_LV_eq, dtype=float)
    C_Mi = np.asarray(C_MiCRM_eq, dtype=float)
    G_surv = C_LV > thresh
    M_surv = C_Mi > thresh
    overlap_mask = G_surv & M_surv
    overlap_count = np.sum(overlap_mask)
    if overlap_count == 0:
        return np.nan, 0
    log_ratios = np.log(C_LV[overlap_mask] / C_Mi[overlap_mask])
    equilibrium_error = np.mean(log_ratios)
    return equilibrium_error, overlap_count


# trajectory abundance deviations 

def err_time_series(times, C_LV_traj, C_Mi_traj, thresh=1e-6):
    mask = (C_LV_traj > thresh) & (C_Mi_traj > thresh)
    overlap_counts = mask.sum(axis=0)
    log_ratios = np.full_like(C_LV_traj, np.nan, dtype=float)
    valid = mask
    log_ratios[valid] = np.log(C_LV_traj[valid] / C_Mi_traj[valid])
    err_t = np.nanmean(log_ratios, axis=0)
    err_t = np.where(np.isnan(err_t), 0.0, err_t)
    return err_t, overlap_counts


def integrate_err(times, err_t):
    valid = ~np.isnan(err_t)
    t_valid = times[valid]
    err_valid = err_t[valid]
    if valid.sum() < 2:
        return np.nan
    integral = np.trapz(err_valid, x=t_valid)
    duration = t_valid[-1] - t_valid[0]
    return integral / duration


# estimate time to equilibrium for trajectory calculations (system reaches equilibrium for both MiCRM + ELVM)

def estimate_teq_traj(times, sol, sol_lv, pT, p_lv, tol=1e-6, window=5):
    T = times.size
    deriv_norms = np.empty((T, 2), dtype=float)
    for j, t in enumerate(times):
        x_mi = sol.y[:, j]
        dx_mi = MiCRM_dxx_numba_wrapper(t, x_mi, pT)
        deriv_norms[j, 0] = norm(dx_mi)
        x_lv = sol_lv.y[:, j]
        dx_lv = LV_dx(x_lv, t, p_lv)
        deriv_norms[j, 1] = norm(dx_lv)
    combined_norm = np.max(deriv_norms, axis=1)
    for j in range(0, T - window + 1):
        if np.all(combined_norm[j : j + window] < tol):
            return j
    return T - 1

# estimate time to equilibrium for timescale calculations (system reaches equilibrium for MiCRM)

def estimate_teq_timescale(times, sol, pT, tol=1e-6, window=5):
    T = times.size
    max_deriv = np.empty(T, dtype=float)
    for j, t in enumerate(times):
        x_mi = sol.y[:, j]
        dx_mi = MiCRM_dxx_numba_wrapper(t, x_mi, pT)
        max_deriv[j] = np.max(np.abs(dx_mi))
    for j in range(T - window + 1):
        if np.all(max_deriv[j : j + window] < tol):
            return j
    return T - 1


# Hessian computation (higher-order interactions)

def F_of_C_jit(C, N, M, u, lam, rho, omega, l, m):
    A = -np.diag(omega)
    for i in range(N):
        for b in range(M):
            Wib = u[i, b] * C[i]
            for a in range(M):
                A[a, b] += l[i, a, b] * Wib
    for a in range(M):
        s = 0.0
        for i in range(N):
            s += u[i, a] * C[i]
        A[a, a] -= s
    R_star = np.linalg.solve(A, rho)
    net = np.empty(N)
    for i in range(N):
        s = 0.0
        for a in range(M):
            s += (1 - lam[i, a]) * u[i, a] * R_star[a]
        net[i] = s - m[i]
    dC = np.empty(N)
    for i in range(N):
        dC[i] = C[i] * net[i]
    return dC


def compute_hessian_norm_nb(C_eq, N, M, u, lam, rho, omega, l, m, eps=1e-6):
    H2_sum = 0.0
    for j in prange(N):
        for k in range(N):
            C_pp = C_eq.copy(); C_pp[j] += eps; C_pp[k] += eps
            C_pm = C_eq.copy(); C_pm[j] += eps; C_pm[k] -= eps
            C_mp = C_eq.copy(); C_mp[j] -= eps; C_mp[k] += eps
            C_mm = C_eq.copy(); C_mm[j] -= eps; C_mm[k] -= eps
            F_pp = F_of_C_jit(C_pp, N, M, u, lam, rho, omega, l, m)
            F_pm = F_of_C_jit(C_pm, N, M, u, lam, rho, omega, l, m)
            F_mp = F_of_C_jit(C_mp, N, M, u, lam, rho, omega, l, m)
            F_mm = F_of_C_jit(C_mm, N, M, u, lam, rho, omega, l, m)
            for i in range(N):
                d2 = (F_pp[i] - F_pm[i] - F_mp[i] + F_mm[i])
                H2_sum += (d2 * d2) / (4 * eps * eps)
    return np.sqrt(H2_sum)


# consumer-resource timescale separation

def timescale_separation_full(J_full, N):
    eigvals, eigvecs = np.linalg.eig(J_full)
    re_times = 1.0 / np.abs(np.real(eigvals))
    weights = np.abs(eigvecs)
    cons_weight = weights[:N, :].sum(axis=0)
    res_weight  = weights[N:, :].sum(axis=0)
    cons_mask = cons_weight >= res_weight
    res_mask  = ~cons_mask
    tau_C = np.min(re_times[cons_mask])
    tau_R = np.max(re_times[res_mask])
    epsilon = tau_C / tau_R
    return tau_C, tau_R, epsilon





#############################################################
########## ACTUAL SIMULATION ################################
#############################################################




outdir = "output" # change this to the desired output directory when running code 
os.makedirs(outdir, exist_ok=True)
paramfile = os.path.join(outdir, "structural_params_all.pkl") # save structural parameters of model 


N = 50 # consumer species number 
M = 25 # resource types number 
L = np.full(N, 0.3) # leakage 
intem = 15 # number of temperatures 
x0 = np.concatenate([np.full(N, 0.1), np.full(M, 1)]) # initial conditions 
temp_vals = np.linspace(273.15+10, 273.15 + 40, intem+1) # temperature values (can change intervals)
rho_t = np.array([-0.5, -0.5]) # thermal generalist-specialist tradeoff 
Tr = 273.15 + 10 # reference temperature for TPC 
Ed = 3.5 # deactivation energy 
rp = 50 # number of replicates 
ttscle = 200 
tint = 1000  

t_eval = np.linspace(0, tint, ttscle) # integration interval 


def run_single_replicate(rep_id):
    structural = generate_params(
        N, M,
        f_u=def_u, f_m=def_m, f_rho=def_rho,
        f_omega=def_omega, f_l=def_l,
        L=L, T=Tr, rho_t=rho_t, Tr=Tr, Ed=Ed,
        rng=default_rng(111 + rep_id)
    )

    metrics = []

    for T in temp_vals:

        # Temperature scaling 
        temp_p, _, _, _ = temp_trait(N, {'T':T,'Tr':Tr,'Ed':Ed,'rho_t':rho_t,'L':L})
        pT = {**structural,
             'u': structural['u']*temp_p[:,0][:,None],
             'm': temp_p[:,1],
             'lambda': structural['l'].sum(axis=2),
             'T': T}
        
        # MiCRM 
        sol = solve_ivp(
            lambda t,y: MiCRM_dxx_numba_wrapper(t,y,pT),
            (0, tint), x0,
            method='LSODA', rtol=1e-4, atol=1e-7,
            t_eval=t_eval)
        
        # ELVM 
        p_lv = eff_LV_params(pT, sol, verbose=False)
        sol_lv = solve_ivp(
            lambda t,y: LV_dx(y,t,p_lv),
            (0, tint), sol.y[:N,0],
            method='LSODA', rtol=1e-4, atol=1e-7,
            t_eval=t_eval)
        
        # Equilibrium values 
        Cmi_eq = sol.y[:N,-1]
        Rmi_eq = sol.y[N:,-1]
        Clv_eq = sol_lv.y[:N,-1]

        # Diversity and abundance deviations 
        ErrEqAb, overlap = err_eq_and_overlap(Clv_eq, Cmi_eq)
        jaccard = jaccard_index(Clv_eq, Cmi_eq, thresh=1e-6)
        sh_mi = shannon(Cmi_eq)
        sh_lv = shannon(Clv_eq)
        bc = bray_curtis_dissimilarity(Clv_eq, Cmi_eq)

        # Trajectory deviation
        err_t,_ = err_time_series(sol.t, sol_lv.y[:N], sol.y[:N])
        j_eq = estimate_teq_traj(sol.t, sol, sol_lv, pT, p_lv, tol=1e-6, window=5)
        ErrTraj = integrate_err(sol.t[:j_eq+1], err_t[:j_eq+1])
        t_eq_index = estimate_teq_timescale(sol.t, sol, pT)
        t_eq_mi = sol.t[t_eq_index]

        # Stability & reactivity
        J_m = MiCRM_jac(pT, sol)
        stab_mi = leading_eigenvalue(J_m)
        react_mi = leading_hermitian_eigenvalue(J_m)

        J_lv_m = eff_LV_jac(p_lv, sol)
        stab_glv = leading_eigenvalue(J_lv_m)
        react_glv = leading_hermitian_eigenvalue(J_lv_m)

        # Timescale separation
        tau_C, tau_R, eps = timescale_separation_full(J_m, N)
        log10_eps_t_eq = np.log10(eps / t_eq_mi)

        # Hessian 
        hnorm = compute_hessian_norm_nb(
            Cmi_eq, N, M,
            pT['u'], pT['lambda'], pT['rho'],
            pT['omega'], pT['l'], pT['m'], eps=1e-6)
        
        # combine metrics 
        metrics.append({
            'replicate': rep_id,
            'T_K': T,
            'T_C': T-273.15,
            'ErrEqAb': ErrEqAb,
            'overlap': overlap,
            'jaccard': jaccard,
            'shannon_mi': sh_mi,
            'shannon_lv': sh_lv,
            'bray_curtis': bc,
            'ErrTraj': ErrTraj,
            't_eq': sol.t[j_eq],
            't_eq_mi': t_eq_mi,
            'stab_mi': stab_mi,
            'stab_glv': stab_glv,
            'abs_stab_err': abs(stab_glv - stab_mi),
            'react_mi': react_mi,
            'react_glv': react_glv,
            'abs_react_err': abs(react_glv - react_mi),
            'tau_C': tau_C,
            'tau_R': tau_R,
            'epsilon': eps,
            'log10_eps_t_eq': log10_eps_t_eq,
            'hessian_norm': hnorm,
            })
    return {'rep_id': rep_id, 'struct': structural, 'record': metrics}


# this code is designed to be run in an HPC, with parallelised processing


if __name__ == '__main__':
    rep_ids = list(range(1, rp+1))
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(run_single_replicate, rep_ids)

    all_metrics = [m for res in results for m in res['record']]
    struct_all = {res['rep_id']: res['struct'] for res in results}

    # save files 
    pd.DataFrame(all_metrics).to_csv(os.path.join(outdir, 'metrics_final.csv'), index=False)
    with open(paramfile, 'wb') as f:
        pickle.dump(struct_all, f)

    print("Simulation complete.")




"""

ACKNOWLEDGEMENTS

Some sections were based on similar code previously developed by members of the Pawar Lab (Danica Duan, Michael Mustri),
such as random generation of temperature parameters, and setting up the MiCRM and ELVM simulations. 


"""