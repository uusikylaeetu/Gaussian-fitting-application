import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

omega_ratio = 1
b_ki = 0.8
# Globaalit epävarmuudet:
sigma_b_ki = 0.1#0.05 * b_ki      # NEW
sigma_omega = 0.05 * omega_ratio       # NEW

from scipy.stats import norm

def yhdista_tau_trunc_bayes(tau, dtau, conf=0.68):
    tau = np.asarray(tau, float)
    dtau = np.asarray(dtau, float)
    w = 1.0 / (dtau**2)
    mu = np.sum(w * tau) / np.sum(w)
    sigma = np.sqrt(1.0 / np.sum(w))

    alpha = (0.0 - mu) / sigma
    cdf_a = norm.cdf(alpha)
    Z = 1.0 - cdf_a

    # Posteriorin moodi = max(0, mu)
    mode = max(0.0, mu)
    # Posteriorin keskiarvo
    post_mean = mu + sigma * norm.pdf(alpha) / Z
    # Trunkeroidun normaalin kvantiilit
    def q(p):
        return mu + sigma * norm.ppf(cdf_a + p * Z)
    lo = max(0.0, q((1 - conf) / 2))
    hi = q((1 + conf) / 2)
    return dict(mu=mu, sigma=sigma, mode=mode, mean=post_mean, lo=lo, hi=hi)

def lue_parametrit(tiedosto):
    params = {}
    with open(tiedosto, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=')
                params[key.strip()] = float(val.strip())
    return params

def painotettu_keskiarvo(tau, dtau):
    weights = 1 / dtau**2
    avg = np.sum(tau * weights) / np.sum(weights)
    err = np.sqrt(1 / np.sum(weights)) 
    return avg, err

def piecewise_model(x, x0, a1, b1, c1, a2, b2):
    x = np.asarray(x)
    y0 = a1 * x0**2 + b1 * x0 + c1
    c2 = y0 - (a2 * x0**2 + b2 * x0)
    y = np.piecewise(
        x,
        [x <= x0, x > x0],
        [lambda x: a1 * x**2 + b1 * x + c1,
         lambda x: a2 * x**2 + b2 * x + c2]
    )
    return y

def käsittele_tiedosto(filename, title, v, del_v, t2, ax_int, ax_tau):
    data = np.loadtxt(filename)

    # KÄYTÄ DATASTA TULEVAA ETÄISYYTTÄ (µm -> m)
    t_flight_60 = data[:, 0] * 1e-6

    Ish_depop   = data[:, 1]
    sig_Ish_dep = data[:, 2]
    Ius_depop   = data[:, 3]
    sig_Ius_dep = data[:, 4]
    Ish_feeder  = data[:, 5]
    sig_Ish_fdr = data[:, 6]
    Ius_feeder  = data[:, 7]
    sig_Ius_fdr = data[:, 8]
    sig_t       = data[:, 9]

    # Päät
    t1 = t_flight_60.min()
    t3 = t_flight_60.max()

    # Sovitus Ish_depop (intensiteettiplotti)
    t_fit_indices1 = np.arange(len(t_flight_60))
    t1_vals = t_flight_60[t_fit_indices1]
    y1_vals = Ish_depop[t_fit_indices1]
    coeffs1 = np.polyfit(t1_vals, y1_vals, 2)
    poly1 = np.poly1d(coeffs1)

    # Oikeanpuoleinen polynomi ankkuroidaan t2:een ja kahteen viimeiseen datapisteeseen
    t3_idx = np.argmin(np.abs(t_flight_60 - t3))
    fixed_y2 = poly1(t2)
    fixed_y3 = Ish_depop[t3_idx]
    t_prev  = t_flight_60[max(t3_idx - 1, 0)]
    y_prev  = Ish_depop[max(t3_idx - 1, 0)]
    fit_x2  = np.array([t2, t_prev, t3])
    fit_y2  = np.array([fixed_y2, y_prev, fixed_y3])
    coeffs2 = np.polyfit(fit_x2, fit_y2, 2)
    poly2 = np.poly1d(coeffs2)

    # Intensiteettiplotti
    # depop (S-komponentti)
    ax_int.errorbar(
        t_flight_60 * 1e6, Ish_depop, yerr=sig_Ish_dep,
        fmt='o', markersize=4, capsize=3, color='tab:orange',
        label=r"$I^{S}_{\mathrm{depop}}$"
    )
    # feeder (S-komponentti)  <-- UUSI
    ax_int.errorbar(
        t_flight_60 * 1e6, Ish_feeder, yerr=sig_Ish_fdr,
        fmt='^', markersize=4, capsize=3, mfc='white', color='tab:blue',
        label=r"$I^{S}_{\mathrm{feeder}}$"
    )  # <-- UUSI

    # Näytetään depop-sovitteen piecewise-käyrät (feederille ei tehdä sovitetta tässä)
    t_fit1 = np.linspace(t1, t2, 300) * 1e6
    t_fit2 = np.linspace(t2, t3, 300) * 1e6
    ax_int.plot(t_fit1, poly1(t_fit1 * 1e-6), color='tab:orange')
    ax_int.plot(t_fit2, poly2(t_fit2 * 1e-6), color='tab:orange', linestyle='-')

    ax_int.set_ylabel(r"$I^{S}$", fontsize=14)  # <-- päivitetty yleisemmäksi
    ax_int.set_xlim(-100, 1400)
    ax_int.set_ylim(0, 1)
  #  ax_int.set_xscale('log')
    ax_int.set_title(f"{title}: intensiteetit")
    ax_int.legend(loc='best', fontsize=9)  # <-- UUSI

    # --- Tau-laskenta (ennallaan) ---
    dpoly1 = poly1.deriv()
    dpoly2 = poly2.deriv()
    #halutut_t = [15e-6, 65e-6, 105e-6, 124e-6, 156e-6]  # 300 µm ja 700 µm
    halutut_t = [5e-6, 270e-6, 700e-6] 
    tau_values = []
    tau_errors = []

    for t_val in halutut_t:
        i = np.argmin(np.abs(t_flight_60 - t_val))
        if t1 <= t_val <= t2:
            dIsh = dpoly1(t_val)
            ddIsh = np.abs(dpoly1.deriv()(t_val) * (t_flight_60[i] - t_val))
        elif t2 < t_val <= t3:
            dIsh = dpoly2(t_val)
            ddIsh = np.abs(dpoly2.deriv()(t_val) * (t_flight_60[i] - t_val))
        else:
            continue

        # Erotus
        y_numer = Ius_depop[i] - b_ki * omega_ratio * Ius_feeder[i]
        print("Ius_depop[i]", Ius_depop[i])
        print("Ius_feeder[i]", Ius_feeder[i])
        print( "b_ki * omega_ratio * Ish_feeder[i]",  b_ki * omega_ratio * Ish_feeder[i])
        # *** VIRHEPROPAGAATIO: mittausepävarmuudet + b_ki ja omega ***
        term_dep   = sig_Ish_dep[i]**2
        term_fdr   = (b_ki * omega_ratio * sig_Ish_fdr[i])**2
        term_b     = (omega_ratio * Ish_feeder[i] * sigma_b_ki)**2
        term_omega = (b_ki * Ish_feeder[i] * sigma_omega)**2
        
                
        dy = np.sqrt(term_dep + term_fdr + term_b + term_omega)

        if y_numer <= 0:
            print(f"VAROITUS: erotus ≤ 0 kohdassa ~{t_val*1e6:.0f} µm (y={y_numer:.4f}). τ voi olla kelvoton.")

        tau = (y_numer / (dIsh * v)) * 1e12  # ps
        print("y_numer", y_numer)
        print("dIsh", dIsh)
        dv_rel = del_v / v
        rel_y   = dy / max(abs(y_numer), 1e-12)
        rel_dI  = abs(ddIsh) / max(abs(dIsh), 1e-12)
        rel_v   = dv_rel

        dtau = np.abs(tau) * np.sqrt(rel_y**2 + rel_dI**2 + rel_v**2)
        tau_values.append(tau)
        tau_errors.append(dtau)



    # --- Tulostetaan yhteenvetotaulukko ---
    print("\n=== Sovitusparametrit ja intensiteettiarvot ===")
    print(f"{'x (µm)':>8} | {'a':>10} | {'b':>10} | {'c':>10} | {'Ius_depop':>12} | {'sig_dep':>10} | {'Ius_feeder':>12} | {'sig_fdr':>10} | {'dIsh':>12}")
    print("-" * 110)

    for t_val in halutut_t:
        i = np.argmin(np.abs(t_flight_60 - t_val))

        # Valitse oikea polynomi-alue
        if t1 <= t_val <= t2:
            a, b, c = coeffs1
            dIsh = dpoly1(t_val)
        else:
            a, b, c = coeffs2
            dIsh = dpoly2(t_val)

        print(f"{t_val*1e6:8.0f} | {a:10.3e} | {b:10.3e} | {c:10.3e} | {Ius_depop[i]:12.4e} | {sig_Ius_dep[i]:10.3e} | {Ius_feeder[i]:12.4e} | {sig_Ius_fdr[i]:10.3e} | {dIsh:12.3e}")

    print("=" * 110 + "\n")




    
    
    
    
    
    tau_values = np.array(tau_values)
    tau_errors = np.array(tau_errors)
    weights = 1 / tau_errors**2
    tau_avg = np.sum(tau_values * weights) / np.sum(weights)
    tau_std = np.sqrt(1 / np.sum(weights)) 

    # Tau-plotti
    x_vals_um = np.array(halutut_t) * 1e6
    ax_tau.errorbar(x_vals_um, tau_values, yerr=tau_errors, fmt='o', color='darkblue', capsize=5)
    stats = yhdista_tau_trunc_bayes(tau_values, tau_errors, conf=0.68)

    # Piirretään moodi-viiva ja 68 % väli; käytä tiedettyä x-aluetta
    x0 = max(1, (t1*1e6)*0.8)
    x1 = (t3*1e6)*1.2
    ax_tau.axhline(stats["mode"], color='black', linestyle='-')
    ax_tau.fill_between([x0, x1], stats["lo"], stats["hi"], color='black', alpha=0.08)
    
    textstr = (
        r"$\hat{{\tau}}_{{\rm MLE}}$ = {mode:.2f} ps"
        "\n"
        r"$68\%$ CI $=[{lo:.2f},\,{hi:.2f}]$ ps"
    ).format(mode=stats["mode"], lo=stats["lo"], hi=stats["hi"])
    
    ax_tau.text(0.98, 0.95, textstr, transform=ax_tau.transAxes,
                fontsize=12, va='top', ha='right')
    

    
    ax_tau.set_ylabel(r'$\tau$ (ps)', fontsize=20)
    ax_tau.set_title(f"{title}: elinaika")
    

    ymin_points = np.min(tau_values - tau_errors)
    ymax_points = np.max(tau_values + tau_errors)
    
    ci_low  = stats["lo"]
    ci_high = stats["hi"]
    
    ymin = min(ci_low, ymin_points)
    ymax = max(ci_high, ymax_points)
    
    span = ymax - ymin
    low  = max(ymin-30, ymin - 0.2 * span)
    high = ymax + 0.4 * span
    
    print("DEBUG ylim:", low, high)
    ax_tau.set_ylim(low, high)

    


    ax_tau.set_xscale('log')



    print("tau =", tau_values.tolist())
    print("tau_errors =", tau_errors.tolist())



    # ----------------------------------------
    # LISÄOMINAISUUS: dIsh_max koko sovitusalueella
    # ----------------------------------------
    def dIsh_fun(x, a, b):
        return 2*a*x + b

    # Vasemman ja oikean sovitealueen derivaatat
    # Käyttöalueet metreinä
    xx_dense = np.linspace(t1, t3, 2000)
    dI_vals = np.zeros_like(xx_dense)

    # Laske dI koko alueella (piecewise)
    for idx, xx in enumerate(xx_dense):
        if xx <= t2:
            a, b, c = coeffs1   # poly1
        else:
            a, b, c = coeffs2   # poly2
        dI_vals[idx] = dIsh_fun(xx, a, b)

    dI_max = np.max(dI_vals)
    x_at_max = xx_dense[np.argmax(dI_vals)] * 1e6   # mikrometreinä

    print("\n=== Maksimi dI_sh_depop sovitusalueella ===")
    print(f"Max dI = {dI_max:.3f} at x = {x_at_max:.1f} µm")
    print("===========================================\n")

    # ---------------------------------------------------------
    # dI arvot nimetyissä pisteissä: 5 µm, 270 µm, 700 µm, 1300 µm
    # ---------------------------------------------------------
    tark_pisteet_um = np.array([5, 270, 700])
    tark_pisteet_m = tark_pisteet_um * 1e-6

    print("=== dI_sh_depop arvot pyydetyissä pisteissä ===")
    print(f"{'x (µm)':>8} | {'2ax+b':>12}")
    print("-" * 30)

    for xp_um, xp_m in zip(tark_pisteet_um, tark_pisteet_m):
        if xp_m <= t2:
            a, b, c = coeffs1
        else:
            a, b, c = coeffs2
        dI_val = dIsh_fun(xp_m, a, b)
        print(f"{xp_um:8.0f} | {dI_val:12.6f}")

    print("=" * 30 + "\n")











    
    return tau_values, tau_errors

def piirra_erotus(filename, title, ax):
    data = np.loadtxt(filename)
    t_flight_60 = data[:, 0] * 1e-6  # m
    Ius_depop   = data[:, 3]
    sig_Ius_dep = data[:, 4]
    Ius_feeder  = data[:, 7]
    sig_Ius_fdr = data[:, 8]

    # Erotus ja sen epävarmuus samoin lisätermein (b, omega)
    erovalinta = (Ius_depop - b_ki*omega_ratio*Ius_feeder)
    sig_erovalinta = np.sqrt(
        sig_Ius_dep**2
        + (b_ki * omega_ratio * sig_Ius_fdr)**2
        + (omega_ratio * Ius_feeder * sigma_b_ki)**2        # NEW
        + (b_ki * Ius_feeder * sigma_omega)**2              # NEW
    )

    ax.errorbar(t_flight_60 * 1e6, erovalinta, yerr=sig_erovalinta, fmt='s', color='gray',
                label=r"$I^{US}_{depop} - b_{ki}\,\omega\, I^{US}_{feeder}$", markersize=3, alpha=0.7)
    ax.set_title(f"{title}: erotus")
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel("Erotus (arb. units)")
    ax.set_xlim(max(1, (t_flight_60.min()*1e6)*0.8), (t_flight_60.max()*1e6)*1.2)
    ax.set_ylim(-0.3, 0.5)
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.tick_params(axis='both', which='both', direction='in')

# --- Ajaminen ---
v = 0.043780 * 3e8
del_v = 61948.481820
t2_ring = 1290e-6

# 1×3-ruudukko (YHDEN TIEDOSTON TARKASTELU)
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for ax in axs.flat:
    ax.tick_params(axis='both', which='both', direction='in')

#tau1, dtau1 = käsittele_tiedosto("int_ring2_66As_3.txt", "Ring 2", v, del_v, t2_ring, axs[0], axs[1])
tau1, dtau1 = käsittele_tiedosto("int_ring2_66As_3_fixed.txt", "Ring 2", v, del_v, t2_ring, axs[0], axs[1])    

#piirra_erotus("int_ring2_66As_3.txt", "Ring 2", axs[2])
piirra_erotus("int_ring2_66As_3_fixed.txt", "Ring 2", axs[2])

    
plt.tight_layout()
plt.show() 
