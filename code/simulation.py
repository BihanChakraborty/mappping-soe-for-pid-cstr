import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import fsolve
import pandas as pd 
 
# Parameters
 
# Kinetics — Garcia-Hernandez et al. (2019) Table 4
k_nc_ref  = 2.50e-5     # L/(mol.s) = m3/(kmol.s) at Tref
k_cat_ref = 9.83e-2     # L/(mol.s) = m3/(kmol.s) at Tref (lumped with K_eq)
Ea_nc     = 46000.0     # J/mol
Ea_cat    = 29700.0     # J/mol
Tref      = 313.15      # K
R         = 8.314       # J/(mol.K)
 
# Acetic acid dissociation: Ka = 1.8e-5 mol/L = 1.8e-5 kmol/m3
Ka        = 1.8e-5      # kmol/m3
# Water concentration (pure, approx constant): 55.5 mol/L = 55.5 kmol/m3
Cw_ref    = 55.5        # kmol/m3
 
# Heat of reaction — DWSIM DIPPR (thermodynamically consistent with NRTL)
# Note: Garcia-Hernandez cites -60 kJ/mol (Zogg); DWSIM gives -50.286 kJ/mol
# DWSIM value used — justified by consistency with SS energy balance
dHrx      = -50286.0    # kJ/kmol
 
# Reactor
V         = 1.0         # m3
Q_vol     = 0.00317316  # m3/s
tau       = V / Q_vol   # s
 
# Feed (nominal)
T0_nom    = 303.0       # K
F_kmol    = 0.1         # kmol/s
Ca0       = (0.16 * F_kmol) / Q_vol   # 5.042 kmol/m3
Cw0       = (0.84 * F_kmol) / Q_vol   # 26.47 kmol/m3
Cp0_feed  = 0.0
 
# Mixture properties — DWSIM back-calculated
Cp_mix    = 97.60       # kJ/(kmol.K)
C_total   = F_kmol / Q_vol   # kmol/m3
 
# Coolant loop
Tc_nom    = 298.0       # K
UA        = 63.24 / (330.0 - Tc_nom)  # 1.976 kW/K
 
# Tc bounds for PID
Tc_min    = 278.0       # K
Tc_max    = 320.0       # K
 
print(f"UA     = {UA:.4f} kW/K")
print(f"tau    = {tau:.1f} s")
print(f"Ca0    = {Ca0:.4f} kmol/m3")
print(f"Cw0    = {Cw0:.4f} kmol/m3")
print()
 
# Kinetic functions
 
def k_nc(T):
    return k_nc_ref * np.exp(-Ea_nc / R * (1/T - 1/Tref))
 
def k_cat(T):
    return k_cat_ref * np.exp(-Ea_cat / R * (1/T - 1/Tref))
 
def H3O_conc(Cp):
    return np.sqrt(Ka * max(Cp, 0.0))
 
def reaction_rate(Ca, Cw, Cp, T):
    Ca  = max(Ca, 0.0)
    Cw  = max(Cw, 1e-9)
    Cp  = max(Cp, 0.0)
    h3o = H3O_conc(Cp)
    k_eff = k_nc(T) + k_cat(T) * h3o / Cw_ref
    return k_eff * Ca * Cw
 
# Steady-state verification
 
def ss_solve(T_set, T0=T0_nom, Ca0_in=None, Cw0_in=None, Tc=Tc_nom):
    if Ca0_in is None: Ca0_in = Ca0
    if Cw0_in is None: Cw0_in = Cw0
 
    def equations(Ca_):
        Ca_ = max(Ca_[0], 1e-9)
        Cp_ = Cp0_feed + 2.0 * (Ca0_in - Ca_)
        Cw_ = Cw0_in - (Ca0_in - Ca_)
        r_  = reaction_rate(Ca_, Cw_, Cp_, T_set)
        return [Ca_ - Ca0_in + tau * r_]
 
    sol   = fsolve(equations, [Ca0_in * 0.7], full_output=True)
    Ca_ss = max(sol[0][0], 0.0)
    Cp_ss = Cp0_feed + 2.0 * (Ca0_in - Ca_ss)
    Cw_ss = Cw0_in - (Ca0_in - Ca_ss)
    r_ss  = reaction_rate(Ca_ss, Cw_ss, Cp_ss, T_set)
    X_ss  = (Ca0_in - Ca_ss) / Ca0_in
    Q_ss  = UA * (T_set - Tc)
    Q_rxn = (-dHrx) * r_ss * V
    Q_sens= F_kmol * Cp_mix * (T_set - T0)
    EB    = Q_rxn - Q_sens - Q_ss
    return Ca_ss, Cw_ss, Cp_ss, r_ss, X_ss, Q_ss, EB
 
print("=" * 60)
print("STEADY-STATE VERIFICATION vs DWSIM")
print("=" * 60)
Ca_ss, Cw_ss, Cp_ss, r_ss, X_ss, Q_ss, EB = ss_solve(330.0)
h3o   = H3O_conc(Cp_ss)
kn    = k_nc(330.0)
kc    = k_cat(330.0)
k_eff = kn + kc * h3o / Cw_ref
 
print(f"Ca (anhydride)     : {Ca_ss:.4f} kmol/m3   | DWSIM: 3.503")
print(f"Cw (water)         : {Cw_ss:.4f} kmol/m3   | DWSIM: 24.933")
print(f"Cp (acetic acid)   : {Cp_ss:.4f} kmol/m3")
print(f"[H3O+]             : {h3o:.4e} kmol/m3")
print(f"k_nc(330K)         : {kn:.4e} m3/(kmol.s)")
print(f"k_cat*[H3O+]/[H2O]: {kc*h3o/Cw_ref:.4e} m3/(kmol.s)")
print(f"k_eff(330K)        : {k_eff:.4e} m3/(kmol.s)")
print(f"Rxn rate           : {r_ss:.5f} kmol/(m3.s)  | DWSIM: 4.887e-3")
print(f"Conversion         : {X_ss*100:.2f}%             | DWSIM: 30.54%")
print(f"Q cooling          : {Q_ss:.2f} kW          | DWSIM: 125.17 kW")
print(f"Energy balance res : {EB:.3f} kW  (target: ~0)")
print(f"Autocatalytic frac : {kc*h3o/Cw_ref/k_eff*100:.1f}% of k_eff")
print()
 
# Safety threshold
f_safety = 0.75
Ts       = 330.0
dTad     = (-dHrx * Ca0) / (C_total * Cp_mix)
Tmax     = Ts + f_safety * dTad
print(f"dTad = {dTad:.2f} K")
print(f"Tmax = {Tmax:.2f} K  (f_safety=0.75)")
print()
 
# ODE system
 
def odes(t, state, Tc, T0, Ca0_in, Cw0_in):
    Ca, Cw, Cp, T = state
    r      = reaction_rate(Ca, Cw, Cp, T)
    Q_cool = UA * (T - Tc)
    dCa    = (Ca0_in - Ca) / tau - r
    dCw    = (Cw0_in - Cw) / tau - r
    dCp    = (0.0    - Cp) / tau + 2.0 * r
    dT     = ((Q_vol / V) * (T0 - T)
              + (-dHrx) * r / (C_total * Cp_mix)
              - Q_cool  / (V * C_total * Cp_mix))
    return np.array([dCa, dCw, dCp, dT])
def numerical_jacobian(f, t, state, args, eps=1e-6):
    state = np.array(state, dtype=float)
    n = len(state)
    J = np.zeros((n, n))
    for i in range(n):
        state_plus = state.copy()
        state_minus = state.copy()
        step = eps * max(abs(state[i]), 1.0)  # scale step to magnitude
        state_plus[i] += step
        state_minus[i] -= step
        f_plus = f(t, state_plus, *args)
        f_minus = f(t, state_minus, *args)
        J[:, i] = (f_plus - f_minus) / (2 * step)
    return J
 
 
ss_state = [2.9944, 24.4241, 4.0958, 330.0]   # [Ca, Cw, Cp, T] at Scenario A SS
ss_args  = (298.0, 330.0, 5.0423, 26.4720)     # (Tc, T0, Ca0_in, Cw0_in)
 
J = numerical_jacobian(odes, 0.0, ss_state, ss_args)
eigenvalues = np.linalg.eigvals(J)
 
print("=" * 60)
print("STABILITY CHECK — Jacobian eigenvalues at steady state")
print("=" * 60)
print(f"Steady state: Ca={ss_state[0]:.4f}  Cw={ss_state[1]:.4f}  "
      f"Cp={ss_state[2]:.4f}  T={ss_state[3]:.2f}K")
print(f"Open-loop conditions: Tc={ss_args[0]}K (fixed, no PID)")
print()
print("Eigenvalues:")
for i, ev in enumerate(eigenvalues):
    real_part = ev.real
    flag = "UNSTABLE" if real_part > 0 else "stable"
    print(f"  lambda_{i+1} = {ev:.6f}   [{flag}]")
print()
if np.any(eigenvalues.real > 0):
    print("RESULT: At least one positive real part -> STEADY STATE IS UNSTABLE.")
else:
    print("RESULT: All real parts negative -> steady state is stable.")
print("=" * 60)
deriv_check = odes(0.0, ss_state, 298.0, 330.0, 5.0423, 26.4720)
print(f"  [diag] derivatives at steady state: dCa={deriv_check[0]:.6e}  dCw={deriv_check[1]:.6e}  dCp={deriv_check[2]:.6e}  dT={deriv_check[3]:.6e}")
 
def ss_solve_coupled(T0, Ca0_in, Cw0_in, Tc, T_guess=None, verbose=False):
    if T_guess is None:
        T_guess = T0  # reasonable starting guess only, not an assumption
 
    def equations(x):
        Ca_, T_ = x
        Ca_ = max(Ca_, 1e-9)
        Cp_ = Cp0_feed + 2.0 * (Ca0_in - Ca_)
        Cw_ = Cw0_in - (Ca0_in - Ca_)
        state = [Ca_, Cw_, Cp_, T_]
        deriv = odes(0.0, state, Tc, T0, Ca0_in, Cw0_in)
        return [deriv[0], deriv[3]]  # [dCa, dT] — both should be ~0
 
    sol = fsolve(equations, [Ca0_in * 0.7, T_guess], full_output=True)
    x_sol = sol[0]
 
    Ca_ss = max(x_sol[0], 0.0)
    T_ss = x_sol[1]
    Cp_ss = Cp0_feed + 2.0 * (Ca0_in - Ca_ss)
    Cw_ss = Cw0_in - (Ca0_in - Ca_ss)
 
    if verbose:
        print(f"  [coupled] Ca_ss={Ca_ss:.4f}  Cw_ss={Cw_ss:.4f}  "
              f"Cp_ss={Cp_ss:.4f}  T_ss={T_ss:.3f}K")
 
    return Ca_ss, Cw_ss, Cp_ss, T_ss
 
 
# COMPARISON — old (decoupled) vs new (coupled) solver
print("=" * 60)
print("COMPARISON: original ss_solve() vs corrected coupled solve")
print("=" * 60)
 
# Original approach: T assumed at 330, mass balance solved for Ca
old_result = ss_solve(T_set=330.0, Ca0_in=5.0423, Cw0_in=26.4720, Tc=298.0)
old_Ca = old_result[0] if isinstance(old_result, tuple) else old_result
print(f"ORIGINAL (T assumed=330.0K): Ca_ss={old_Ca:.4f}")
 
# Corrected approach: T solved jointly with Ca, given feed temp=330, Tc=298
new_Ca, new_Cw, new_Cp, new_T = ss_solve_coupled(
    T0=303.0, Ca0_in=5.0423, Cw0_in=26.4720, Tc=298.0, verbose=True
)
print(f"CORRECTED: Ca_ss={new_Ca:.4f}  T_ss={new_T:.3f}K  "
      f"(vs assumed T=330.0K, delta={new_T-330.0:+.3f}K)")
print()
print(f"Ca delta (corrected - original): {new_Ca - old_Ca:+.4f} kmol/m3")
print("=" * 60)
def check_multiplicity(T0, Ca0_in, Cw0_in, Tc, T_guesses):
    print("=" * 60)
    print(f"MULTIPLICITY CHECK at Tc={Tc}K, T0={T0}K")
    print("=" * 60)
    results = []
    for Tg in T_guesses:
        Ca_ss, Cw_ss, Cp_ss, T_ss = ss_solve_coupled(
            T0=T0, Ca0_in=Ca0_in, Cw0_in=Cw0_in, Tc=Tc, T_guess=Tg
        )
        conv = (Ca0_in - Ca_ss) / Ca0_in * 100
        results.append((Tg, Ca_ss, T_ss, conv))
        print(f"  guess T={Tg:6.1f}K  ->  Ca_ss={Ca_ss:.4f}  "
              f"T_ss={T_ss:7.3f}K  conv={conv:5.1f}%")
 
    print()
    distinct_T = sorted(set(round(r[2], 1) for r in results))
    print(f"Distinct converged T_ss values found: {distinct_T}")
    if len(distinct_T) > 1:
        print("RESULT: Multiple steady states confirmed — reactor is bistable")
        print("        (or multistable) at this Tc.")
    else:
        print("RESULT: All guesses converged to the same point — no")
        print("        multiplicity detected at this Tc with these guesses.")
    print("=" * 60)
    return results
 
 
# Sweep a wide range of initial temperature guesses, low to high,
# to see whether fsolve lands on more than one fixed point.
guesses = [295, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
check_multiplicity(T0=330.0, Ca0_in=5.0423, Cw0_in=26.4720, Tc=298.0,
                    T_guesses=guesses)
def ss_solve_coupled_checked(T0, Ca0_in, Cw0_in, Tc, T_guess=None,
                               tol=1e-4, verbose=False):
    if T_guess is None:
        T_guess = T0
 
    def equations(x):
        Ca_, T_ = x
        Ca_ = max(Ca_, 1e-9)
        Cp_ = Cp0_feed + 2.0 * (Ca0_in - Ca_)
        Cw_ = Cw0_in - (Ca0_in - Ca_)
        state = [Ca_, Cw_, Cp_, T_]
        deriv = odes(0.0, state, Tc, T0, Ca0_in, Cw0_in)
        return [deriv[0], deriv[3]]
 
    sol, infodict, ier, msg = fsolve(
        equations, [Ca0_in * 0.7, T_guess], full_output=True
    )
 
    Ca_ss = max(sol[0], 0.0)
    T_ss = sol[1]
 
    # Re-check residuals AND physical plausibility, not just fsolve's
    # own ier flag (ier=1 can still hide a bad answer at low tolerance)
    residuals = equations([Ca_ss, T_ss])
    max_resid = max(abs(r) for r in residuals)
 
    physically_valid = (0.0 <= Ca_ss <= Ca0_in)
    converged = (ier == 1) and (max_resid < tol) and physically_valid
 
    Cp_ss = Cp0_feed + 2.0 * (Ca0_in - Ca_ss)
    Cw_ss = Cw0_in - (Ca0_in - Ca_ss)
 
    if verbose:
        status = "VALID" if converged else "REJECTED"
        print(f"  guess T={T_guess:6.1f}K -> Ca_ss={Ca_ss:.4f}  "
              f"T_ss={T_ss:8.3f}K  max_resid={max_resid:.2e}  [{status}]"
              + ("" if converged else f"  ({msg.strip()})"))
 
    return Ca_ss, Cw_ss, Cp_ss, T_ss, converged
 
 
def check_multiplicity_verified(T0, Ca0_in, Cw0_in, Tc, T_guesses):
    print("=" * 60)
    print(f"VERIFIED MULTIPLICITY CHECK at Tc={Tc}K, T0={T0}K")
    print("=" * 60)
    valid_results = []
    for Tg in T_guesses:
        Ca_ss, Cw_ss, Cp_ss, T_ss, ok = ss_solve_coupled_checked(
            T0=T0, Ca0_in=Ca0_in, Cw0_in=Cw0_in, Tc=Tc,
            T_guess=Tg, verbose=True
        )
        if ok:
            valid_results.append(T_ss)
 
    print()
    print(f"Guesses that converged to a physically valid root: "
          f"{len(valid_results)} of {len(T_guesses)}")
 
    if valid_results:
        distinct_T = sorted(set(round(t, 0) for t in valid_results))
        print(f"Distinct VALID T_ss values: {distinct_T}")
        if len(distinct_T) > 1:
            print("RESULT: Multiple genuinely converged steady states "
                  "-> bistability CONFIRMED.")
        else:
            print("RESULT: Only one steady state converges validly -> "
                  "NOT bistable. Earlier scatter was solver failure, "
                  "not real physics.")
    else:
        print("RESULT: No guesses converged validly. Something is wrong "
              "with the solve setup itself.")
    print("=" * 60)
 
 
guesses = [295, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390]
check_multiplicity_verified(T0=303.0, Ca0_in=5.0423, Cw0_in=26.4720,
                              Tc=298.0, T_guesses=guesses)
# Runge-Kutta integration
 
def rk4(f, t, y, dt, *args):
    k1 = f(t,        y,              *args)
    k2 = f(t + dt/2, y + dt/2 * k1, *args)
    k3 = f(t + dt/2, y + dt/2 * k2, *args)
    k4 = f(t + dt,   y + dt   * k3, *args)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
 
# PID controller
 
class PID:
    def __init__(self, Kp, Ki, Kd, mv_min, mv_max, sp, bias, integral_init=0.0):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd
        self.mv_min=mv_min; self.mv_max=mv_max
        self.sp=sp; self.bias=bias
        self.integral=integral_init
        self.prev_err=None
 
    def update(self, meas, dt):
        err   = meas - self.sp
        deriv = (err - self.prev_err)/dt if self.prev_err is not None else 0.0
        self.prev_err = err
        mv_raw = self.bias - (self.Kp*err + self.Ki*self.integral + self.Kd*deriv)
        mv_sat = np.clip(mv_raw, self.mv_min, self.mv_max)
        if (mv_sat==mv_raw) or \
           (mv_sat==self.mv_max and err>0) or \
           (mv_sat==self.mv_min and err<0):
            self.integral += err * dt
        return mv_sat, err
 
# PID tuning
Kp = 4.3
Ki = 0.05
Kd = 50.0
dt = 2.0    # s
 
# Simulation functions
 
def simulate(t_end, T_sp, T0_fn, Ca0_fn, Cw0_fn, Tc_bias,
             state_ic, Kp, Ki, Kd):
    t_arr = np.arange(0.0, t_end+dt, dt)
    n = len(t_arr)
    rec = {k: np.zeros(n) for k in ['Ca','Cw','Cp','T','Tc','Q','X','err']}
    state = state_ic.copy()
    pid = PID(Kp, Ki, Kd, Tc_min, Tc_max, T_sp, Tc_bias)
    pid.prev_err = state_ic[3] - T_sp
    for i, t in enumerate(t_arr):
        Ca,Cw,Cp,T = state
        T0n=T0_fn(t); Ca0n=Ca0_fn(t); Cw0n=Cw0_fn(t)
        Tc_now, err = pid.update(T, dt)
        rec['Ca'][i]=Ca; rec['Cw'][i]=Cw; rec['Cp'][i]=Cp
        rec['T'][i]=T;   rec['Tc'][i]=Tc_now
        rec['Q'][i]=UA*(T-Tc_now)
        rec['X'][i]=max(0,(Ca0n-Ca)/Ca0n)
        rec['err'][i]=err
        state = rk4(odes, t, state, dt, Tc_now, T0n, Ca0n, Cw0n)
    return t_arr, rec
 
def simulate_vary_bias(t_end, T_sp, T0_fn, Ca0_fn, Cw0_fn, Tc_bias_fn,
                       state_ic, Kp, Ki, Kd):
    t_arr = np.arange(0.0, t_end+dt, dt)
    n = len(t_arr)
    rec = {k: np.zeros(n) for k in ['Ca','Cw','Cp','T','Tc','Q','X','err']}
    state = state_ic.copy()
    pid = PID(Kp, Ki, Kd, Tc_min, Tc_max, T_sp, Tc_nom)
    pid.prev_err = state_ic[3] - T_sp
    for i, t in enumerate(t_arr):
        Ca,Cw,Cp,T = state
        T0n=T0_fn(t); Ca0n=Ca0_fn(t); Cw0n=Cw0_fn(t)
        pid.bias = Tc_bias_fn(t)
        Tc_now, err = pid.update(T, dt)
        rec['Ca'][i]=Ca; rec['Cw'][i]=Cw; rec['Cp'][i]=Cp
        rec['T'][i]=T;   rec['Tc'][i]=Tc_now
        rec['Q'][i]=UA*(T-Tc_now)
        rec['X'][i]=max(0,(Ca0n-Ca)/Ca0n)
        rec['err'][i]=err
        state = rk4(odes, t, state, dt, Tc_now, T0n, Ca0n, Cw0n)
    return t_arr, rec
 
# Scenario A: setpoint step 330 K -> 340 K
 
print("Running Scenario A: Setpoint tracking 330K -> 340K ...")
IC_ss    = np.array([Ca_ss, Cw_ss, Cp_ss, 330.0])
T0_c     = lambda t: T0_nom
Ca0_c    = lambda t: Ca0
Cw0_c    = lambda t: Cw0
 
t1,r1 = simulate(500.0, 330.0, T0_c, Ca0_c, Cw0_c, Tc_nom, IC_ss, Kp,Ki,Kd)
IC2   = np.array([r1['Ca'][-1],r1['Cw'][-1],r1['Cp'][-1],r1['T'][-1]])
t2,r2 = simulate(2500.0,340.0, T0_c, Ca0_c, Cw0_c, Tc_nom, IC2,  Kp,Ki,Kd)
 
t_A = np.concatenate([t1, t2+500])
T_A = np.concatenate([r1['T'],  r2['T']])
Tc_A= np.concatenate([r1['Tc'], r2['Tc']])
Q_A = np.concatenate([r1['Q'],  r2['Q']])
X_A = np.concatenate([r1['X'],  r2['X']])
Cp_A= np.concatenate([r1['Cp'], r2['Cp']])
 
print(f"  Settled T  : {T_A[-1]:.3f} K  (target 340 K)")
print(f"  SS offset  : {T_A[-1]-340:.3f} K")
print(f"  Settled Tc : {Tc_A[-1]:.3f} K")
print(f"  Settled Q  : {Q_A[-1]:.2f} kW")
print(f"  Conversion : {X_A[-1]*100:.2f}%")
print(f"  Max T      : {T_A.max():.2f} K  (Tmax={Tmax:.1f}K): {'SAFE' if T_A.max()<Tmax else 'UNSAFE'}")
print()
 
# Scenario B: coolant disturbance +10 K
 
print("Running Scenario B: Coolant supply temp 298K -> 308K at t=500s ...")
Tc_bias_B = lambda t: Tc_nom if t < 500.0 else 308.0
 
t_B,r_B = simulate_vary_bias(3000.0, 330.0, T0_c, Ca0_c, Cw0_c,
                              Tc_bias_B, IC_ss, Kp,Ki,Kd)
 
print(f"  Settled T  : {r_B['T'][-1]:.3f} K  (target 330 K)")
print(f"  SS offset  : {r_B['T'][-1]-330:.3f} K")
print(f"  Settled Tc : {r_B['Tc'][-1]:.3f} K")
print(f"  Settled Q  : {r_B['Q'][-1]:.2f} kW")
print(f"  Conversion : {r_B['X'][-1]*100:.2f}%")
print(f"  Max T      : {r_B['T'].max():.2f} K  (Tmax={Tmax:.1f}K): {'SAFE' if r_B['T'].max()<Tmax else 'UNSAFE'}")
print()
 
# Plots
 
fig = plt.figure(figsize=(15, 11))
fig.suptitle(
    "Stage 2 — PID-Controlled Exothermic CSTR  |  Autocatalytic Kinetics\n"
    "IIT Guwahati  |  Garcia-Hernandez et al. (2019)",
    fontsize=13, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(4, 2, hspace=0.52, wspace=0.35)
 
cT='#C0392B'; cTc='#E67E22'; cQ='#2471A3'; cX='#1E8449'
cSP='#717D7E'; cTmax='#922B21'; cCp='#8E44AD'
 
def vline(ax, x):
    ax.axvline(x/60, color='k', lw=0.8, ls='--', alpha=0.5)
 
# A: Temperature
ax=fig.add_subplot(gs[0,0])
ax.plot(t_A/60,T_A, color=cT,  lw=1.8, label='Reactor T')
ax.plot(t_A/60,Tc_A,color=cTc, lw=1.2, ls='--', label='Coolant Tc')
ax.axhline(330,color=cSP,lw=1,ls=':',label='SP=330K')
ax.axhline(340,color=cSP,lw=1,ls='--',label='SP=340K')
ax.axhline(Tmax,color=cTmax,lw=1.2,ls='-.',label=f'Tmax={Tmax:.0f}K')
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Temperature (K)',title='A — Setpoint Step: Temperature')
ax.legend(fontsize=7); ax.grid(True,alpha=0.25)
 
# A: Q cooling
ax=fig.add_subplot(gs[1,0])
ax.plot(t_A/60,Q_A,color=cQ,lw=1.8)
ax.axhline(Q_ss,color=cSP,lw=1,ls=':',label=f'SS={Q_ss:.0f}kW')
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Q cooling (kW)',title='A — Setpoint Step: Cooling Duty')
ax.legend(fontsize=7); ax.grid(True,alpha=0.25)
 
# A: Conversion
ax=fig.add_subplot(gs[2,0])
ax.plot(t_A/60,X_A*100,color=cX,lw=1.8)
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Conversion (%)',title='A — Setpoint Step: Conversion')
ax.grid(True,alpha=0.25)
 
# A: Acetic acid
ax=fig.add_subplot(gs[3,0])
ax.plot(t_A/60,Cp_A*1000,color=cCp,lw=1.8)
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Acetic acid (mol/m3)',title='A — Setpoint Step: Acetic Acid')
ax.grid(True,alpha=0.25)
 
# B: Temperature
ax=fig.add_subplot(gs[0,1])
ax.plot(t_B/60,r_B['T'], color=cT,  lw=1.8, label='Reactor T')
ax.plot(t_B/60,r_B['Tc'],color=cTc, lw=1.2, ls='--', label='Coolant Tc')
ax.axhline(330,color=cSP,lw=1,ls=':',label='SP=330K')
ax.axhline(Tmax,color=cTmax,lw=1.2,ls='-.',label=f'Tmax={Tmax:.0f}K')
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Temperature (K)',title='B — Coolant Disturbance: Temperature')
ax.legend(fontsize=7); ax.grid(True,alpha=0.25)
 
# B: Q cooling
ax=fig.add_subplot(gs[1,1])
ax.plot(t_B/60,r_B['Q'],color=cQ,lw=1.8)
ax.axhline(Q_ss,color=cSP,lw=1,ls=':',label=f'SS={Q_ss:.0f}kW')
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Q cooling (kW)',title='B — Coolant Disturbance: Cooling Duty')
ax.legend(fontsize=7); ax.grid(True,alpha=0.25)
 
# B: Conversion
ax=fig.add_subplot(gs[2,1])
ax.plot(t_B/60,r_B['X']*100,color=cX,lw=1.8)
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Conversion (%)',title='B — Coolant Disturbance: Conversion')
ax.grid(True,alpha=0.25)
 
# B: Acetic acid
ax=fig.add_subplot(gs[3,1])
ax.plot(t_B/60,r_B['Cp']*1000,color=cCp,lw=1.8)
vline(ax,500)
ax.set(xlabel='Time (min)',ylabel='Acetic acid (mol/m3)',title='B — Coolant Disturbance: Acetic Acid')
ax.grid(True,alpha=0.25)
 
plt.savefig("stage2_v2_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot saved.")
 
# Summary
print()
print("="*60)
print("STAGE 2 FINAL SUMMARY")
print("="*60)
print(f"Kinetics    : Full autocatalytic — Garcia-Hernandez 2019")
print(f"              r = Ca*Cw*(k_nc + k_cat'*[H3O+]/[H2O])")
print(f"Coolant     : UA={UA:.3f} kW/K, Tc_nom={Tc_nom} K")
print(f"PID gains   : Kp={Kp}, Ki={Ki}, Kd={Kd}")
print(f"Safety      : Tmax={Tmax:.2f} K (dTad={dTad:.1f}K, f=0.75)")
print()
print(f"Autocatalytic fraction at SS: {kc*h3o/Cw_ref/k_eff*100:.1f}% of k_eff")
print()
print("Scenario A — Setpoint step 330->340K:")
print(f"  Final T    : {T_A[-1]:.3f} K  |  Offset: {T_A[-1]-340:.3f} K")
print(f"  Max T      : {T_A.max():.2f} K  -> {'SAFE' if T_A.max()<Tmax else 'UNSAFE'}")
print(f"  Conversion : {X_A[-1]*100:.2f}%")
print()
print("Scenario B — Coolant disturbance +10K:")
print(f"  Final T    : {r_B['T'][-1]:.3f} K  |  Offset: {r_B['T'][-1]-330:.3f} K")
print(f"  Max T      : {r_B['T'].max():.2f} K  -> {'SAFE' if r_B['T'].max()<Tmax else 'UNSAFE'}")
print(f"  Conversion : {r_B['X'][-1]*100:.2f}%")
print("="*60)
print("\nStage 2 complete. Proceed to Stage 3.")
# SOE classifier
# Kp, Ki, Kd, dt, tau, Tmax, C_total
 
SOE_DURATION = 10 * tau     # ~3151.7 s — 10x residence time, catches slow
                            # oscillatory instability a shorter window would miss
SOE_SETPOINT = 330.0        # K — nominal operating setpoint for the classifier.
                            # needs to become a 4th argument, not a fixed constant.
 
N_RECOVERY = 5              # recovery-period multiplier on tau_cl. Standard
                            # process-control settling-time convention uses
                            # 3-5x time constant for a linear first-order system
                            # to reach 95-99% recovery (Seborg/Ogunnaike & Ray).
                            # N=5 is the conservative end, justified here because
                            # the closed loop is nonlinear and PID-controlled,
                            # not a plain first-order response.
                            # defines a divergence-based runaway criterion with no
                            # lit review before submission.
TAU_CL = tau                # closed-loop time constant, taken as reactor
RECOVERY_WINDOW = N_RECOVERY * TAU_CL   # ~1575.8 s
 
 
def classify_point(T0, Ca0, Tc_supply, Cw0=None, verbose=False):
    if Cw0 is None:
        Cw0 = C_total - Ca0   # PLACEHOLDER, replace with correct mass balance

    T0_fn  = lambda t: T0
    Ca0_fn = lambda t: Ca0
    Cw0_fn = lambda t: Cw0

    # Initialize the REACTOR at its own true coupled steady state for this
    # operating point (mass AND energy balance), not at raw feed
    # concentration, and not by assuming reactor T = feed T.
    Ca_ss, Cw_ss, Cp_ss, T_ss, ss_converged = ss_solve_coupled_checked(
        T0=T0, Ca0_in=Ca0, Cw0_in=Cw0, Tc=Tc_supply
    )
    Ca_ss = max(Ca_ss, 0.0)

    if not ss_converged:
        print(f"  [WARNING] steady-state solve did not converge for "
              f"T0={T0}, Ca0={Ca0}, Tc_supply={Tc_supply}")

    print(f"  [diag] Ca_ss={Ca_ss:.4f}  Cw_ss={Cw_ss:.4f}  Cp_ss={Cp_ss:.4f}  T_ss={T_ss:.3f}")
    state_ic = [Ca_ss, Cw_ss, Cp_ss, T_ss]   # [Ca, Cw, Cp, T], true steady-state

    t_arr_sim, rec = simulate(
        t_end=SOE_DURATION,
        T_sp=SOE_SETPOINT,
        T0_fn=T0_fn, Ca0_fn=Ca0_fn, Cw0_fn=Cw0_fn,
        Tc_bias=Tc_supply,
        state_ic=state_ic,
        Kp=Kp, Ki=Ki, Kd=Kd
    )
    print(f"  [diag] T[0:5]={rec['T'][:5]}")
    print(f"  [diag] Tc[0:5]={rec['Tc'][:5]}")

    peak_T = float(np.max(rec['T']))
    T_arr = rec['T']

    window_steps = int(5 * 60 / dt)
    settled_T = float(np.mean(T_arr[-window_steps:]))

    conversion = float(1 - rec['Ca'][-1] / Ca0)

    # --- Two-condition runaway test, per project definition ---
    above = T_arr > Tmax
    label = "SAFE"
    max_excursion_duration = 0.0

    if np.any(above):
        diffs = np.diff(above.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if above[0]:
            starts = np.r_[0, starts]
        if above[-1]:
            ends = np.r_[ends, len(above)]

        for s, e in zip(starts, ends):
            duration = t_arr_sim[e - 1] - t_arr_sim[s]
            max_excursion_duration = max(max_excursion_duration, duration)
            if duration > RECOVERY_WINDOW:
                label = "UNSAFE"
                break

    if verbose:
        print(f"T0={T0:6.1f}K  Ca0={Ca0:6.3f}  Tc_supply={Tc_supply:6.1f}K  "
              f"-> peakT={peak_T:7.2f}K  settledT={settled_T:7.2f}K  "
              f"maxExcursion={max_excursion_duration:6.1f}s "
              f"(limit={RECOVERY_WINDOW:.1f}s)  "
              f"conv={conversion*100:5.1f}%  [{label}]")

    return {
        'T0': T0, 'Ca0': Ca0, 'Tc_supply': Tc_supply,
        'peak_T': peak_T, 'settled_T': settled_T,
        'max_excursion_s': max_excursion_duration,
        'conversion': conversion, 'label': label
    }
 
if __name__ == "__main__":
    print(f"SOE_DURATION = {SOE_DURATION:.1f} s ({SOE_DURATION/tau:.1f} x tau)")
    print(f"Tmax         = {Tmax:.2f} K\n")
 
   # Check 1: verify classify_point() reproduces the validated Stage 2
    # steady state at nominal operating conditions. NOTE: this is NOT the
    # same as Stage 2's Scenario A (which tested a setpoint STEP 330->340K).
    # classify_point() holds a fixed SOE_SETPOINT=330K throughout, so the
    # correct expectation here is that it settles near 330K with conversion
    # matching the Stage 1 DWSIM baseline (~40.6%), not Scenario A's post-step
    print("Check 1 — reproducing Stage 2 nominal steady state:")
    classify_point(T0=303.0, Ca0=5.0423, Tc_supply=298.0, verbose=True)
    # NOTE: Ca0=None will break — you need the actual Stage 2 initial Ca0 here
 
    # cooling capacity can't keep up. Confirms the classifier actually catches
    # UNSAFE cases and isn't just always returning SAFE.
    print("\nCheck 2 — deliberately pushing toward UNSAFE:")
    classify_point(T0=340.0, Ca0=5.0423, Tc_supply=319.0, verbose=True)

    # SOE sweep
    import csv
    import itertools

    print("\n" + "=" * 60)
    print("STAGE 4 — SWEEPING SAFE OPERATING ENVELOPE")
    print("=" * 60)

    # Sweep ranges (9 points each, 729 total combinations)
    T0_range        = np.linspace(293.0, 350.0, 9)   # feed temperature, K
    Ca0_range       = np.linspace(3.8, 6.3, 9)         # feed anhydride conc, kmol/m3
    Tc_supply_range = np.linspace(288.0, 320.0, 9)     # coolant supply temp, K

    total_points = len(T0_range) * len(Ca0_range) * len(Tc_supply_range)
    print(f"Grid: {len(T0_range)} x {len(Ca0_range)} x {len(Tc_supply_range)} "
          f"= {total_points} points")
    print("This will take a while — each point runs a full dynamic simulation.\n")

    results = []
    count = 0

    for T0_val, Ca0_val, Tc_val in itertools.product(T0_range, Ca0_range, Tc_supply_range):
        count += 1
        result = classify_point(T0=T0_val, Ca0=Ca0_val, Tc_supply=Tc_val, verbose=False)
        results.append(result)
        if count % 25 == 0 or count == total_points:
            print(f"  [{count}/{total_points}] T0={T0_val:.1f}K Ca0={Ca0_val:.3f} "
                  f"Tc={Tc_val:.1f}K -> {result['label']}")

    # --- Write to CSV ---
    csv_path = "soe_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'T0', 'Ca0', 'Tc_supply', 'peak_T', 'settled_T',
            'max_excursion_s', 'conversion', 'label'
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    n_safe = sum(1 for r in results if r['label'] == 'SAFE')
    n_unsafe = total_points - n_safe
    print(f"\nSweep complete: {n_safe} SAFE, {n_unsafe} UNSAFE out of {total_points}")
    print(f"Results written to {csv_path}")
   
    # SOE visualization
  

    print("\nGenerating SOE envelope plots...")
    df = pd.read_csv(csv_path)

    # Pick 3 representative Ca0 slices: low, nominal, high
    ca0_unique = sorted(df['Ca0'].unique())
    ca0_low  = ca0_unique[0]
    ca0_mid  = ca0_unique[len(ca0_unique) // 2]
    ca0_high = ca0_unique[-1]
    ca0_slices = [ca0_low, ca0_mid, ca0_high]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Safe Operating Envelope — T0 vs Tc_supply, at fixed Ca0",
                 fontsize=13, fontweight='bold')

    for ax, ca0_val in zip(axes, ca0_slices):
        sub = df[np.isclose(df['Ca0'], ca0_val)]
        safe = sub[sub['label'] == 'SAFE']
        unsafe = sub[sub['label'] == 'UNSAFE']

        ax.scatter(safe['Tc_supply'], safe['T0'], c='#1E8449', marker='s',
                   s=180, label='SAFE', alpha=0.85)
        ax.scatter(unsafe['Tc_supply'], unsafe['T0'], c='#C0392B', marker='s',
                   s=180, label='UNSAFE', alpha=0.85)

        ax.set_xlabel('Coolant supply Tc (K)')
        ax.set_ylabel('Feed temperature T0 (K)')
        ax.set_title(f'Ca0 = {ca0_val:.3f} kmol/m3')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(r"C:\Users\bihan\soe_envelope_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to C:\\Users\\bihan\\soe_envelope_plot.png")
    # Monte Carlo simulation and probabilistic safety mapping
    print("\n" + "=" * 60)
    print("STAGE 6 — MONTE CARLO PROBABILISTIC SAFETY MAPPING")
    print("=" * 60)

    np.random.seed(42)   # reproducibility — same random draws every run
    N_SAMPLES = 500

    # Distributions (mean, std), clipped to physical/PID bounds
    T0_mean, T0_std       = 303.0, 15.0
    Ca0_mean, Ca0_std     = 5.0423, 0.8
    Tc_mean, Tc_std       = 298.0, 8.0

    T0_samples  = np.clip(np.random.normal(T0_mean, T0_std, N_SAMPLES), 280.0, 360.0)
    Ca0_samples = np.clip(np.random.normal(Ca0_mean, Ca0_std, N_SAMPLES), 1.0, C_total)
    Tc_samples  = np.clip(np.random.normal(Tc_mean, Tc_std, N_SAMPLES), Tc_min, Tc_max)

    print(f"Sampling {N_SAMPLES} scenarios from:")
    print(f"  T0        ~ Normal({T0_mean}, {T0_std}) K, clipped [280, 360]")
    print(f"  Ca0       ~ Normal({Ca0_mean}, {Ca0_std}) kmol/m3, clipped [1.0, {C_total:.2f}]")
    print(f"  Tc_supply ~ Normal({Tc_mean}, {Tc_std}) K, clipped [{Tc_min}, {Tc_max}]")
    print("This will take a while...\n")

    mc_results = []
    for i in range(N_SAMPLES):
        result = classify_point(
            T0=T0_samples[i], Ca0=Ca0_samples[i], Tc_supply=Tc_samples[i],
            verbose=False
        )
        mc_results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == N_SAMPLES:
            print(f"  [{i+1}/{N_SAMPLES}] -> {result['label']}")

    # --- Write Monte Carlo results to CSV ---
    mc_csv_path = r"C:\Users\bihan\monte_carlo_results.csv"
    with open(mc_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'T0', 'Ca0', 'Tc_supply', 'peak_T', 'settled_T',
            'max_excursion_s', 'conversion', 'label'
        ])
        writer.writeheader()
        for r in mc_results:
            writer.writerow(r)

    n_mc_safe = sum(1 for r in mc_results if r['label'] == 'SAFE')
    p_safe = n_mc_safe / N_SAMPLES
    print(f"\nMonte Carlo complete: {n_mc_safe}/{N_SAMPLES} SAFE")
    print(f"Estimated probability of safe operation (given these uncertainty "
          f"distributions): {p_safe*100:.1f}%")
    print(f"Results written to {mc_csv_path}")

    # --- Probabilistic Safety Map: bin into T0 vs Tc_supply grid, ---
    # --- color by P(SAFE) within each bin ---
    mc_df = pd.DataFrame(mc_results)
    mc_df['safe_flag'] = (mc_df['label'] == 'SAFE').astype(int)

    n_bins = 8
    T0_bins = np.linspace(T0_samples.min(), T0_samples.max(), n_bins + 1)
    Tc_bins = np.linspace(Tc_samples.min(), Tc_samples.max(), n_bins + 1)

    prob_grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (mc_df['T0'] >= T0_bins[i]) & (mc_df['T0'] < T0_bins[i+1]) &
                (mc_df['Tc_supply'] >= Tc_bins[j]) & (mc_df['Tc_supply'] < Tc_bins[j+1])
            )
            if mask.sum() > 0:
                prob_grid[i, j] = mc_df.loc[mask, 'safe_flag'].mean()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(prob_grid, origin='lower', aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1,
                   extent=[Tc_bins[0], Tc_bins[-1], T0_bins[0], T0_bins[-1]])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('P(SAFE)')
    ax.set_xlabel('Coolant supply Tc (K)')
    ax.set_ylabel('Feed temperature T0 (K)')
    ax.set_title(f'Probabilistic Safety Map (N={N_SAMPLES} Monte Carlo samples)\n'
                f'Overall P(SAFE) = {p_safe*100:.1f}%')
    plt.tight_layout()
    plt.savefig(r"C:\Users\bihan\probabilistic_safety_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Probabilistic safety map saved to "
          "C:\\Users\\bihan\\probabilistic_safety_map.png")
    
    # Heat-of-reaction sensitivity
    # DIPPR value (-50,286 kJ/kmol) vs Garcia-Hernandez et al. (2019)
    # cited value (-60,000 kJ/kmol). Both flow through dTad -> Tmax,
    # and dHrx also enters the energy balance in odes() directly, so
    # this rerun captures both the shifted threshold AND the shifted
    # dynamics, not just a static recalculation.
    print("\n" + "=" * 60)
    print("STAGE 4b — SENSITIVITY OF SOE TO HEAT OF REACTION")
    print("=" * 60)

    # --- Store the DIPPR-based sweep already computed above as baseline ---
    dHrx_dippr = dHrx           # -50286.0, whatever ran above
    Tmax_dippr = Tmax
    dTad_dippr = dTad
    results_dippr = results     # the 729-point list from Stage 4 above
    n_safe_dippr = n_safe
    n_unsafe_dippr = n_unsafe

    print(f"Baseline (DIPPR)  dHrx={dHrx_dippr:.1f} kJ/kmol  "
          f"dTad={dTad_dippr:.2f} K  Tmax={Tmax_dippr:.2f} K  "
          f"-> {n_safe_dippr}/{total_points} SAFE ({n_safe_dippr/total_points*100:.1f}%)")

    # --- Switch to the literature value and recompute the threshold ---
    dHrx = -60000.0
    dTad = (-dHrx * Ca0) / (C_total * Cp_mix)
    Tmax = Ts + f_safety * dTad
    print(f"Alternate (lit.)  dHrx={dHrx:.1f} kJ/kmol  "
          f"dTad={dTad:.2f} K  Tmax={Tmax:.2f} K")
    print("Rerunning the full 729-point grid sweep with the literature value...\n")

    results_lit = []
    count = 0
    for T0_val, Ca0_val, Tc_val in itertools.product(T0_range, Ca0_range, Tc_supply_range):
        count += 1
        result = classify_point(T0=T0_val, Ca0=Ca0_val, Tc_supply=Tc_val, verbose=False)
        results_lit.append(result)
        if count % 25 == 0 or count == total_points:
            print(f"  [{count}/{total_points}] -> {result['label']}")

    n_safe_lit = sum(1 for r in results_lit if r['label'] == 'SAFE')
    n_unsafe_lit = total_points - n_safe_lit

    # --- Write comparison CSV ---
    sens_csv_path = r"C:\Users\bihan\hrx_sensitivity_results.csv"
    with open(sens_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['T0', 'Ca0', 'Tc_supply', 'label_dippr', 'label_lit', 'flipped'])
        for r_d, r_l in zip(results_dippr, results_lit):
            flipped = r_d['label'] != r_l['label']
            writer.writerow([r_d['T0'], r_d['Ca0'], r_d['Tc_supply'],
                              r_d['label'], r_l['label'], flipped])

    n_flipped = sum(1 for r_d, r_l in zip(results_dippr, results_lit)
                     if r_d['label'] != r_l['label'])

    print("\n" + "=" * 60)
    print("ΔHrx SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"DIPPR   (-50,286 kJ/kmol): Tmax={Tmax_dippr:.2f} K -> "
          f"{n_safe_dippr}/{total_points} SAFE ({n_safe_dippr/total_points*100:.1f}%)")
    print(f"Literature (-60,000 kJ/kmol): Tmax={Tmax:.2f} K -> "
          f"{n_safe_lit}/{total_points} SAFE ({n_safe_lit/total_points*100:.1f}%)")
    print(f"Points that changed classification: {n_flipped}/{total_points} "
          f"({n_flipped/total_points*100:.1f}%)")
    print(f"Results written to {sens_csv_path}")
    print("=" * 60)

    # Restore DIPPR values as the script's "operating" state, since that's
    dHrx = dHrx_dippr
    dTad = dTad_dippr
    Tmax = Tmax_dippr

    # STAGE 6b — ΔHrx SENSITIVITY CHECK FOR MONTE CARLO
    # Reuses the SAME 500 random draws (T0_samples, Ca0_samples,
    # Tc_samples) already generated above under seed 42, so this is a
    # like-for-like comparison of the same scenarios under two ΔHrx
    # values, not a fresh random sample.
    print("\n" + "=" * 60)
    print("STAGE 6b — SENSITIVITY OF MONTE CARLO SAFETY TO HEAT OF REACTION")
    print("=" * 60)

    mc_results_dippr = mc_results     # the 500-point list from Stage 6 above
    n_mc_safe_dippr = n_mc_safe
    p_safe_dippr = p_safe

    print(f"Baseline (DIPPR)  dHrx={dHrx_dippr:.1f} kJ/kmol  Tmax={Tmax_dippr:.2f} K  "
          f"-> {n_mc_safe_dippr}/{N_SAMPLES} SAFE ({p_safe_dippr*100:.1f}%)")

    # --- Switch to the literature value and recompute the threshold ---
    dHrx = -60000.0
    dTad = (-dHrx * Ca0) / (C_total * Cp_mix)
    Tmax = Ts + f_safety * dTad
    print(f"Alternate (lit.)  dHrx={dHrx:.1f} kJ/kmol  dTad={dTad:.2f} K  Tmax={Tmax:.2f} K")
    print("Rerunning the same 500 Monte Carlo scenarios with the literature value...\n")

    mc_results_lit = []
    for i in range(N_SAMPLES):
        result = classify_point(
            T0=T0_samples[i], Ca0=Ca0_samples[i], Tc_supply=Tc_samples[i],
            verbose=False
        )
        mc_results_lit.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == N_SAMPLES:
            print(f"  [{i+1}/{N_SAMPLES}] -> {result['label']}")

    n_mc_safe_lit = sum(1 for r in mc_results_lit if r['label'] == 'SAFE')
    p_safe_lit = n_mc_safe_lit / N_SAMPLES

    # --- Write comparison CSV ---
    mc_sens_csv_path = r"C:\Users\bihan\mc_hrx_sensitivity_results.csv"
    with open(mc_sens_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['T0', 'Ca0', 'Tc_supply', 'label_dippr', 'label_lit', 'flipped'])
        for r_d, r_l in zip(mc_results_dippr, mc_results_lit):
            flipped = r_d['label'] != r_l['label']
            writer.writerow([r_d['T0'], r_d['Ca0'], r_d['Tc_supply'],
                              r_d['label'], r_l['label'], flipped])

    n_mc_flipped = sum(1 for r_d, r_l in zip(mc_results_dippr, mc_results_lit)
                        if r_d['label'] != r_l['label'])

    print("\n" + "=" * 60)
    print("MONTE CARLO ΔHrx SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"DIPPR      (-50,286 kJ/kmol): Tmax={Tmax_dippr:.2f} K -> "
          f"{n_mc_safe_dippr}/{N_SAMPLES} SAFE ({p_safe_dippr*100:.1f}%)")
    print(f"Literature (-60,000 kJ/kmol): Tmax={Tmax:.2f} K -> "
          f"{n_mc_safe_lit}/{N_SAMPLES} SAFE ({p_safe_lit*100:.1f}%)")
    print(f"Scenarios that changed classification: {n_mc_flipped}/{N_SAMPLES} "
          f"({n_mc_flipped/N_SAMPLES*100:.1f}%)")
    print(f"Results written to {mc_sens_csv_path}")
    print("=" * 60)

    # Restore DIPPR values as the script's final operating state
    dHrx = dHrx_dippr
    dTad = dTad_dippr
    Tmax = Tmax_dippr