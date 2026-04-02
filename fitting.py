import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, Button, CheckButtons

from scipy.optimize import curve_fit
import os
saved_fits = {}



_updating_mu1_box = False
_updating_mu2_box = False
_updating_a_box = False
_updating_b_box = False

def sync_auto_checkboxes(label):
    global _updating_auto_checkbox

    target_A1 = auto_fit_params[label]["A1"]
    target_A2 = auto_fit_params[label]["A2"]
    target_sigma1 = auto_fit_params[label]["sigma1"]
    target_sigma2 = auto_fit_params[label]["sigma2"]
    
    target_mu1 = auto_fit_params[label]["mu1"]
    target_mu2 = auto_fit_params[label]["mu2"]

    current_status = auto_checkbox.get_status()

    _updating_auto_checkbox = True
    try:
        if current_status[0] != target_A1:
            auto_checkbox.set_active(0)
        if current_status[1] != target_A2:
            auto_checkbox.set_active(1)
        if current_status[2] != target_sigma1:
            auto_checkbox.set_active(2)
        if current_status[3] != target_sigma2:
            auto_checkbox.set_active(3)
        if current_status[4] != target_mu1:
            auto_checkbox.set_active(4)
        if current_status[5] != target_mu2:
            auto_checkbox.set_active(5)
    finally:
        _updating_auto_checkbox = False
        
        
        
_updating_auto_checkbox = False

def parse_saved_params(filename):
    """Palauttaa sanakirjan tallennetuista sovitusparametreista tiedostosta."""
    params = {}
    if not os.path.exists(filename):
        return params

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            label = parts[0]
            try:
                A1 = float(parts[1])
                mu1 = float(parts[2])
                sigma1 = float(parts[3])
                A2 = float(parts[4])
                mu2 = float(parts[5])
                sigma2 = float(parts[6])
                a = float(parts[7])
                b = float(parts[8])
                params[label] = {
                    "A1": A1,
                    "mu1": mu1,
                    "sigma1": sigma1,
                    "A2": A2,
                    "mu2": mu2,
                    "sigma2": sigma2,
                    "a": a,
                    "b": b
                }
            except:
                continue
    return params


def on_radio(label):
    global current_label, _updating_link
    current_label = label
    sync_auto_checkboxes(label)

    # 1) Vaihda ranget ensin
    update_slider_ranges(label)

    # 2) Lataa tallennetut parametrit
    if "decay" in label:
        saved_params = parse_saved_params("params_decay_ring2_66As_8.txt")
    elif "feeder" in label:
        saved_params = parse_saved_params("params_feeder_ring2_66As_8.txt")
    else:
        saved_params = {}

    # 3) Aseta arvot turvallisesti (ei käynnistetä linkkauksia kesken kaiken)
    def set_safe(slider, val):
        global _updating_link
        _updating_link = True
        try:
            slider.set_val(val)
        finally:
            _updating_link = False

    if label in saved_params:
        sp = saved_params[label]  # hae tallennetut arvot ensin
    
        # tallenna amplitude-parametrit oikein
        amp_params[label] = (sp["A1"], sp["A2"])
    
        # päivitä kaikki sliderit
        set_safe(slider_mu1,    sp["mu1"])
        set_safe(slider_sigma1, sp["sigma1"])
        set_safe(slider_mu2,    sp["mu2"])
        set_safe(slider_sigma2, sp["sigma2"])
        set_safe(slider_a,      sp["a"])
        set_safe(slider_b,      sp["b"])
        set_safe(slider_A1,     sp["A1"])
        set_safe(slider_A2,     sp["A2"])


    else:
        # fallback: käytä files[label] p0-arvoja, ettei jäädytä edellisen labelin arvoihin
        p0 = files[label][2]
        set_safe(slider_mu1, p0[1])
        set_safe(slider_sigma1, p0[2])
        set_safe(slider_mu2, p0[4])
        set_safe(slider_sigma2, p0[5])
        a_val, b_val = ab_params.get(label, (p0[-2], p0[-1]))
        
        set_safe(slider_a, a_val)
        set_safe(slider_b, b_val)

    # 4) Feeder-lukitus näkyviin heti


    update_plot()
    
    # Lataa A1 ja A2 tälle spektrille
    A1_val, A2_val = amp_params[label]

    set_safe(slider_A1, A1_val)
    set_safe(slider_A2, A2_val)




# --- Globaalit funktiot ---
# --- Yhteenvetokuva kaikista spektri+fit -pareista ---------------------------

def _get_saved_param_dicts():
    decay_dict  = parse_saved_params("params_decay_ring2_66As_8.txt")
    feeder_dict = parse_saved_params("params_feeder_ring2_66As_8.txt")
    return decay_dict, feeder_dict

def _params_for_label(label):
    """Hae labelille mu/sigma/a/b: ensisijaisesti tallennetusta tiedostosta,
    muuten fit_results-popteista. Feederissä varmistetaan mu1 = mu2 - DELTA_MU,
    jollei tallennettua arvoa ole."""
    decay_dict, feeder_dict = _get_saved_param_dicts()
    source = feeder_dict if "feeder" in label else decay_dict
    if label in source:
        return source[label]

    res = fit_results.get(label)
    if res is None:
        return None
    _, _, _, popt = res
    d = {
        "mu1":   float(popt[1]),
        "sigma1":float(popt[2]),
        "mu2":   float(popt[4]),
        "sigma2":float(popt[5]),
        "a":     float(popt[6]),
        "b":     float(popt[7]),
    }


def _amplitudes_given_fixed(x, y, mu1, s1, mu2, s2, a, b):
    """Sovita A1 ja A2 kiinnitetyillä mu/sigma/a/b-arvoilla, jotta käyrä vastaa dataa."""
    def m(xx, A1, A2):
        return double_gaussian_with_linear(xx, A1, mu1, s1, A2, mu2, s2, a, b)
    try:
        popt, _ = curve_fit(m, x, y, p0=[max(y), max(y)],
                            bounds=([0.0, 0.0], [np.inf, np.inf]))
        return float(popt[0]), float(popt[1])
    except Exception:
        return 0.0, 0.0

def draw_all_spectra(event=None):
    rows = ["5um", "270um", "700um", "1300um"]
    grid = [(f"{r}_decay", f"{r}_feeder") for r in rows]

    # True  = taustasubtrahoidut (kuten aiemmin)
    # False = raakadata + tausta mukana
    PLOT_BG_SUB = True

    fig2, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=False, sharey=False)

    decay_saved = parse_saved_params("params_decay_ring2_66As_8.txt")
    feeder_saved = parse_saved_params("params_feeder_ring2_66As_8.txt")

    for r, (lab_decay, lab_feeder) in enumerate(grid):
        for c, label in enumerate([lab_decay, lab_feeder]):
            ax2 = axes[r, c]
            res = fit_results.get(label)

            # Etäisyys otsikkoon (rivi kertoo sen jo)
            dist_um = rows[r].replace("um", "")
            if c == 0:
                trans = r"$2^+ \to 0^+$"
                ax2.set_ylabel(r"Counts / 1 keV")
            else:
                trans = r"$3^+ \to 2^+$"
                ax2.set_ylabel("")
            title = rf"$x = {dist_um}\,\mu\mathrm{{m}}$ ({trans})"
            ax2.set_title(title)

            if res is None:
                ax2.text(0.5, 0.5, "ei dataa", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_axis_off()
                continue

            x_fit, y_fit, x_smooth, full_popt = res

            saved_dict = feeder_saved if "feeder" in label else decay_saved

            if label in saved_dict:
                sp = saved_dict[label]
                A1 = max(0.0, float(sp["A1"]))
                A2 = max(0.0, float(sp["A2"]))

                mu2 = float(sp["mu2"])
                sigma1 = float(sp["sigma1"])
                sigma2 = float(sp["sigma2"])
                a = float(sp["a"])
                b = float(sp["b"])


            else:
                A1_f, mu1_f, sigma1_f, A2_f, mu2_f, sigma2_f, a_f, b_f = full_popt

                A1 = max(0.0, float(A1_f))
                A2 = max(0.0, float(A2_f))

                mu2 = float(mu2_f)
                sigma1 = float(sigma1_f)
                sigma2 = float(sigma2_f)
                a = float(a_f)
                b = float(b_f)



            # Komponentit (ilman taustaa)
            g1 = A1 * np.exp(-(x_smooth - mu1) ** 2 / (2.0 * sigma1 ** 2))
            g2 = A2 * np.exp(-(x_smooth - mu2) ** 2 / (2.0 * sigma2 ** 2))

            bg_smooth = a * x_smooth + b
            bg_fit = a * x_fit + b

            if PLOT_BG_SUB:
                y_plot = y_fit - bg_fit
                total_plot = g1 + g2

                ax2.step(x_fit, y_plot, where="mid", lw=1)
                ax2.plot(x_smooth, g1, lw=1)
                ax2.plot(x_smooth, g2, lw=1)
                ax2.plot(x_smooth, total_plot, lw=1.4)
                ax2.axhline(0.0, ls="--", lw=1, color="k")

                ymin = min(np.min(y_plot), -0.5)
                ymax = max(np.max(y_plot), np.max(total_plot)) * 1.15 + 0.5
                ax2.set_ylim(ymin, ymax)
            else:
                total_model = g1 + g2 + bg_smooth

                ax2.step(x_fit, y_fit, where="mid", lw=1)
                ax2.plot(x_smooth, g1 + bg_smooth, lw=1)
                ax2.plot(x_smooth, g2 + bg_smooth, lw=1)
                ax2.plot(x_smooth, bg_smooth, ls="--", lw=1, color="k")
                ax2.plot(x_smooth, total_model, lw=1.4)

                ymin = min(np.min(y_fit), 0.0)
                ymax = max(np.max(y_fit), np.max(total_model)) * 1.15 + 0.5
                ax2.set_ylim(ymin, ymax)

            # --- Akselit: x-label vain alimmalle riville ---
            if r == len(rows) - 1:
                ax2.set_xlabel(r"$E\,[\mathrm{keV}]$")
                ax2.tick_params(axis="x", labelbottom=True)
            else:
                ax2.set_xlabel("")
                ax2.tick_params(axis="x", labelbottom=False)

            # y-label: "Counts / 1 keV"
            
            ax2.tick_params(axis="both", direction="in")
                     
	   
    fig2.tight_layout()
    fig2.show()




def double_gaussian_with_linear(x, A1, mu1, sigma1, A2, mu2, sigma2, a, b):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return g1 + g2 + a * x + b

def single_gaussian_with_linear(x, A1, mu1, sigma1, a, b):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    return g1 + a * x + b

def calculate_reduced_chi2(x, y, y_fit, dof):
    # Vältä nollien neliöjuurta
    sigma = np.sqrt(np.maximum(y, 1))
    chi2 = np.sum(((y - y_fit) / sigma)**2)
    return chi2 / dof


def write_int_ring2():
    label_map = {
        "5um": 5,
        "270um": 270,
        "700um": 700,
        "1300um": 1300
    }

    last_column = {
        5: 0.1,
        270: 0.2,
        700: 0.3,
        1300: 0.4,
    }

    data_decay = {}
    data_feeder = {}

    with open("ratios_only_ring2_66As_8.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            label = parts[0]
            values = list(map(float, parts[1:]))
            for prefix, key in label_map.items():
                if label.startswith(f"{prefix}_decay"):
                    data_decay[key] = values
                elif label.startswith(f"{prefix}_feeder"):
                    data_feeder[key] = values

    with open("int_ring2_66As_8.fit", "w") as f_out:
        for key in sorted(label_map.values()):
            row = [key]
            row.extend(data_decay.get(key, [0.0]*4))
            row.extend(data_feeder.get(key, [0.0]*4))
          #  row.append(last_column.get(key, 0.0))
            f_out.write(" ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + "\n")



def estimate_background_simple(x, y):
    a = 0
    b = np.min(y)
    return a, b


def calculate_ratio_and_error(A1, sigma1, A2, sigma2, dA1, dA2, dsigma1, dsigma2):
    """Laskee tarkat alueet ja niiden suhteet virhearvioineen."""
    Area1 = A1 * sigma1 * np.sqrt(2 * np.pi)
    Area2 = A2 * sigma2 * np.sqrt(2 * np.pi)

    # Virheiden propagointi
  ##  dArea1 = Area1 * np.sqrt((dA1 / A1)**2 + (dsigma1 / sigma1)**2)
  #  dArea2 = Area2 * np.sqrt((dA2 / A2)**2 + (dsigma2 / sigma2)**2)

    dArea1 = np.sqrt(Area1)
    dArea2 = np.sqrt(Area2)
    
    
    total = Area1 + Area2
    ratio1 = Area1 / total
    ratio2 = Area2 / total

    def safe_ratio_error(area, darea):
        return 0 if area <= 0 else darea / area

    dRatio1 = ratio1 * np.sqrt(
        safe_ratio_error(Area1, dArea1)**2 + safe_ratio_error(Area2, dArea2)**2
    )
 
    dRatio2 = ratio2 * np.sqrt(
        safe_ratio_error(Area1, dArea1)**2 + safe_ratio_error(Area2, dArea2)**2
    )


    return Area1, dArea1, Area2, dArea2, ratio1, dRatio1, ratio2, dRatio2



def overwrite_entry_in_file(filename, label, line_content, ratio_info=None):
    """Korvaa tiedostossa rivin, joka alkaa annetulla labelilla, uudella rivillä.
    Tulostaa lisäksi tallennettavat arvot, jos ratio_info annetaan."""

    lines = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            lines = f.readlines()
    
    # Pidä kaikki rivit, jotka eivät ala samalla labelilla
    lines = [line for line in lines if not line.strip().startswith(label)]

    # Lisää uusi rivi
    lines.append(line_content)

    # Kirjoita kaikki rivit takaisin tiedostoon
    with open(filename, "w") as f:
        f.writelines(lines)

    print(f"↪️ Korvattiin rivi tiedostossa {filename} labelilla: {label}")
    print(f"📝 Uusi rivi:\n{line_content.strip()}")

    if ratio_info:
        ratio1, dratio1, ratio2, dratio2 = ratio_info
        print("✅ Tiedostoon tallennettavat arvot:")
        print(f"{label:<14} {ratio1:9.4f} {dratio1:8.4f} {ratio2:9.4f} {dratio2:8.4f}")



# --- Tallennus ---
def export_saved_fits_to_file(filename="ratios_ring2_66As_8.txt"):
    if not saved_fits:
        print("❌ Ei sovituksia tallennettavaksi tiedostoon.")
        return

    header = (
        f"{'Label':<18} {'A1':>6} {'mu1':>7} {'σ1':>6} {'A2':>6} {'mu2':>7} {'σ2':>6} "
        f"{'a':>6} {'b':>6} {'Area1':>8} {'Area2':>8} {'Ratio1':>7} {'dRatio1':>8} {'Ratio2':>7} {'dRatio2':>8}\n"
    )

    lines = [header, "-" * len(header) + "\n"]

    for label, params in saved_fits.items():
        A1, mu1, sigma1, A2, mu2, sigma2, a, b, Area1, Area2 = params

        try:
            x_synthetic = np.linspace(mu1 - 5 * sigma1, mu2 + 5 * sigma2, 100)
            y_synthetic = double_gaussian_with_linear(x_synthetic, A1, mu1, sigma1, A2, mu2, sigma2, a, b)
            _, cov = curve_fit(
                double_gaussian_with_linear,
                x_synthetic, y_synthetic,
                p0=[A1, mu1, sigma1, A2, mu2, sigma2, a, b],
                bounds=([0, 0, 1.0, 0, 0, 1.0, -np.inf, -np.inf],
                        [np.inf, np.inf, 8.0, np.inf, np.inf, 8.0, np.inf, np.inf])
            )
            sigma1_err = np.sqrt(cov[2, 2])
            sigma2_err = np.sqrt(cov[5, 5])
            mu1_err = np.sqrt(cov[1, 1])
            mu2_err = np.sqrt(cov[4, 4])
        except:
            sigma1_err = 0.1
            sigma2_err = 0.1

        _, dArea1, _, dArea2, ratio1, dratio1, ratio2, dratio2 = calculate_ratio_and_error(
            A1, sigma1, A2, sigma2, sigma1_err, sigma2_err
        )


        _, dArea1, _, dArea2, ratio1, dratio1, ratio2, dratio2 = calculate_ratio_and_error(
            A1, sigma1, A2, sigma2, sigma1_err, sigma2_err
        )
        print("sigma1_err", sigma1_err)
        print("sigma2_err", sigma2_err)
        print("mu1_err", mu1_err)
        print("mu2_err", mu2_err)



        

        lines.append(
            f"{label:<18} {A1:6.0f} {mu1:7.2f} {sigma1:6.2f} "
            f"{A2:6.0f} {mu2:7.2f} {sigma2:6.2f} {a:6.2f} {b:6.1f} "
            f"{Area1:8.1f} {Area2:8.1f} {ratio1:7.4f} {dRatio1:8.4f} {ratio2:7.4f} {dRatio2:8.4f}"
        )

    with open(filename, "w") as f:
        f.writelines(lines)
    print(f"✅ Sovitukset kirjoitettu tiedostoon: {filename}")

def is_feeder_label(label: str) -> bool:
    return "feeder" in label




min_decay = 910
max_decay = 1010
min_feed = 330
max_feed = 429
# tiedostot ja alkuarvot
# tiedostot ja alkuarvot
files = {
    "5um_decay": ("d35um_379_JG3.dat", (min_decay, max_decay), [10, 953.4482655124717, 1.8, 10, 963.5164297894413, 1.8, 0, 0]),
    "270um_decay": ("d300um_379_01.dat", (min_decay, max_decay), [10, 953.4482655124717, 1.8, 10, 963.5164297894413, 1.8, 0, 0]),
    "700um_decay": ("d700um_379_01.dat", (min_decay, max_decay), [10, 953.4482655124717, 1.8, 10, 963.5164297894413, 1.8, 0, 0]),
    "1300um_decay": ("d1300um_379.dat", (min_decay, max_decay), [10, 953.4482655124717, 1.8, 10, 963.5164297894413, 1.8, 0, 0]),
  
    "5um_feeder": ("d35um_379_JG3.dat", (min_feed, max_feed), [20, 375.4482655124717, 1.4, 20, 378.5164297894413, 1.4, 0, 0]),
    "270um_feeder": ("d300um_379_01.dat", (min_feed, max_feed), [20, 375.4482655124717, 1.4, 20, 378.5164297894413, 1.4, 0, 0]),
    "700um_feeder": ("d700um_379_01.dat", (min_feed, max_feed), [20, 375.7944469668705, 1.4, 20, 378.944321665719,  1.4, 0, 0]),
    "1300um_feeder": ("d1300um_379.dat", (min_feed, max_feed), [20, 375.7944469668705, 1.4, 20, 378.944321665719,  1.4, 0, 0])
 
}

fit_results = {}

_updating_link = False  # estää rekursiivisen päivityssilmukan


for label, (filename, fit_range, p0) in files.items():
    if not os.path.exists(filename):
        fit_results[label] = None
        continue
    data = np.loadtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    mask = (x >= fit_range[0]) & (x <= fit_range[1])
    x_fit = x[mask]
    y_fit = y[mask]
    try:
        # 🔽 TÄHÄN LISÄÄ TÄMÄ:
        # Laske a ja b automaattisesti reunojen perusteella
        a_bg, b_bg = estimate_background_simple(x_fit, y_fit)

        p0[-2] = a_bg
        p0[-1] = b_bg

        lower_bounds = [0, 0, 1.0, 0, 0, 1.0, -np.inf, -np.inf]
        upper_bounds = [np.inf, np.inf, 5.0, np.inf, np.inf, 5.0, np.inf, np.inf]
        y_fit_for_fit = np.maximum(y_fit, 0)
        popt, _ = curve_fit(
            double_gaussian_with_linear, x_fit, y_fit, p0=p0,
            bounds=(lower_bounds, upper_bounds)
        )

        x_smooth = np.linspace(x_fit[0], x_fit[-1], 1000)
        fit_results[label] = (x_fit, y_fit, x_smooth, popt)
    except Exception as e:
        print(f"Sovitus epäonnistui kohteelle {label}: {e}")
        fit_results[label] = None


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.3, bottom=0.35)


radio_ax = plt.axes([0.01, 0.4, 0.2, 0.5])
radio = RadioButtons(radio_ax, list(files.keys()))





checkbox_ax = plt.axes([0.01, 0.25, 0.2, 0.1])

auto_checkbox = CheckButtons(
    checkbox_ax,
    [
        r'Auto-fit $A_1$',
        r'Auto-fit $A_2$',
        r'Auto-fit $\sigma_1$',
        r'Auto-fit $\sigma_2$',
        r'Auto-fit $\mu_1$',
        r'Auto-fit $\mu_2$'
    ],
    [True, True, False, False, False, False]
)


auto_fit_params = {
    label: {
        "A1": True,
        "A2": True,
        "sigma1": False,
        "sigma2": False,
        "mu1": False,
        "mu2": False
    }
    for label in files
}
_updating_auto_checkbox = False

def auto_checkbox_callback(label):
    global current_label, _updating_auto_checkbox

    if _updating_auto_checkbox:
        return

    if label == r'Auto-fit $A_1$':
        auto_fit_params[current_label]["A1"] = not auto_fit_params[current_label]["A1"]
    elif label == r'Auto-fit $A_2$':
        auto_fit_params[current_label]["A2"] = not auto_fit_params[current_label]["A2"]
    elif label == r'Auto-fit $\sigma_1$':
        auto_fit_params[current_label]["sigma1"] = not auto_fit_params[current_label]["sigma1"]
    elif label == r'Auto-fit $\sigma_2$':
        auto_fit_params[current_label]["sigma2"] = not auto_fit_params[current_label]["sigma2"]
    elif label == r'Auto-fit $\mu_1$':
        auto_fit_params[current_label]["mu1"] = not auto_fit_params[current_label]["mu1"]
    elif label == r'Auto-fit $\mu_2$':
        auto_fit_params[current_label]["mu2"] = not auto_fit_params[current_label]["mu2"]

    update_plot()

auto_checkbox.on_clicked(auto_checkbox_callback)




def get_sigma_uncertainties(label, default_sigma1=0.1, default_sigma2=0.1):
    """
    Hae sigma1- ja sigma2-epävarmuudet saved_fits:stä, jos saatavilla.
    Muuten käytä oletusarvoja.
    """
    if label in saved_fits:
        params = saved_fits[label]
        # saved_fits[label] = [A1, mu1, sigma1, A2, mu2, sigma2, a, b,
        #                      Area1, Area2, sigma1_err, sigma2_err, ...]
        sigma1_err = float(params[10])
        sigma2_err = float(params[11])
        return sigma1_err, sigma2_err

    return default_sigma1, default_sigma2









slider_ax_a = plt.axes([0.3, 0.15, 0.55, 0.03])
slider_a = Slider(slider_ax_a, 'a (bg)', -1.5, 1.5, valinit=0)

slider_ax_b = plt.axes([0.3, 0.1, 0.55, 0.03])
slider_b = Slider(slider_ax_b, 'b (bg)', -2000, 2000, valinit=10)


slider_ax_A1 = plt.axes([0.88, 0.75, 0.1, 0.03])
slider_ax_A2 = plt.axes([0.88, 0.70, 0.1, 0.03])

slider_A1 = Slider(slider_ax_A1, r'$A_1$ (amp)', 0, 30, valinit=10)
slider_A2 = Slider(slider_ax_A2, r'$A_2$ (amp)', 0, 30, valinit=10)


from matplotlib.widgets import TextBox

# Luo tekstikenttä b-arvolle
textbox_ax_b = plt.axes([0.92, 0.1, 0.1, 0.03])  # sijoittelu sliderin oikealle puolelle
textbox_b = TextBox(textbox_ax_b, '', initial=str(slider_b.val))



_updating_a_box = False
_updating_b_box = False

def submit_b(text):
    global _updating_b_box
    if _updating_b_box:
        return
    try:
        value = float(text)
        value = np.clip(value, slider_b.valmin, slider_b.valmax)
        if abs(slider_b.val - value) > 1e-12:
            slider_b.set_val(value)
    except ValueError:
        pass

def on_slider_b_change(val):
    global _updating_b_box
    if _updating_b_box:
        return
    _updating_b_box = True
    try:
        textbox_b.text_disp.set_text(f"{val:.4f}")
    finally:
        _updating_b_box = False
    update_plot()




textbox_ax_a = plt.axes([0.92, 0.15, 0.1, 0.03])  # sijoittelu sliderin oikealle puolelle
textbox_a = TextBox(textbox_ax_a, '', initial=str(slider_a.val))

_updating_a_box = False
_updating_b_box = False

def submit_a(text):
    global _updating_a_box
    if _updating_a_box:
        return
    try:
        value = float(text)
        value = np.clip(value, slider_a.valmin, slider_a.valmax)
        if abs(slider_a.val - value) > 1e-12:
            slider_a.set_val(value)
    except ValueError:
        pass

def on_slider_a_change(val):
    global _updating_a_box
    if _updating_a_box:
        return
    _updating_a_box = True
    try:
        textbox_a.text_disp.set_text(f"{val:.4f}")
    finally:
        _updating_a_box = False
    update_plot()


def on_slider_A1_change(val):
    amp_params[current_label] = (val, amp_params[current_label][1])
    update_plot()

def on_slider_A2_change(val):
    amp_params[current_label] = (amp_params[current_label][0], val)
    update_plot()

slider_A1.on_changed(on_slider_A1_change)
slider_A2.on_changed(on_slider_A2_change)



slider_ax_mu1 = plt.axes([0.3, 0.2, 0.55, 0.03])
slider_mu1 = Slider(slider_ax_mu1, r'$\mu_1$ (keV)', 940, 970, valinit=953)

slider_ax_mu2 = plt.axes([0.3, 0.25, 0.55, 0.03])
slider_mu2 = Slider(slider_ax_mu2, r'$\mu_2$ (keV)', 940, 970, valinit=961)

textbox_ax_mu1 = plt.axes([0.92, 0.2, 0.55, 0.03])  # sijoittelu sliderin oikealle puolelle
textbox_mu1 = TextBox(textbox_ax_mu1, '', initial=str(slider_mu1.val))

textbox_ax_mu2 = plt.axes([0.92, 0.25, 0.55, 0.03])
textbox_mu2 = TextBox(textbox_ax_mu2, '', initial=str(slider_mu2.val))


_updating_mu1_box = False

def submit_mu1(text):
    global _updating_mu1_box
    if _updating_mu1_box:
        return
    try:
        value = float(text)
        value = np.clip(value, slider_mu1.valmin, slider_mu1.valmax)
        if abs(slider_mu1.val - value) > 1e-12:
            slider_mu1.set_val(value)
    except ValueError:
        pass

def on_slider_mu1_change(val):
    global _updating_mu1_box
    if _updating_mu1_box:
        return

    _updating_mu1_box = True
    try:
        textbox_mu1.text_disp.set_text(f"{val:.4f}")
    finally:
        _updating_mu1_box = False

    update_plot()

_updating_mu2_box = False

def submit_mu2(text):
    global _updating_mu2_box
    if _updating_mu2_box:
        return
    try:
        value = float(text)
        value = np.clip(value, slider_mu2.valmin, slider_mu2.valmax)
        if abs(slider_mu2.val - value) > 1e-12:
            slider_mu2.set_val(value)
    except ValueError:
        pass
        
        


textbox_b.on_submit(submit_b)
textbox_a.on_submit(submit_a)
textbox_mu1.on_submit(submit_mu1)
textbox_mu2.on_submit(submit_mu2)

def on_slider_mu2_change(val):
    global _updating_mu2_box, _updating_link
    if _updating_link or _updating_mu2_box:
        return

    _updating_mu2_box = True
    try:
        textbox_mu2.text_disp.set_text(f"{val:.4f}")
    finally:
        _updating_mu2_box = False

    update_plot()

fig.canvas.draw_idle()

slider_ax_sigma2 = plt.axes([0.3, 0.94, 0.55, 0.03])  # ylin
slider_sigma2 = Slider(slider_ax_sigma2, r'$\sigma_2$', 1.0, 6.0, valinit=2.5)

slider_ax_sigma1 = plt.axes([0.3, 0.89, 0.55, 0.03])  # sen alla
slider_sigma1 = Slider(slider_ax_sigma1, r'$\sigma_1$', 1.0, 6.0, valinit=2.5)

current_label = list(files.keys())[0]

f = 1.3
def custom_chi2_segment(x_fit, y_fit, model_func, label, mu1, sigma1, mu2, sigma2):
    """Valitsee sopivan segmentin chi²-laskentaan piikkien perusteella."""
    if "5um" in label:
        # Käytä vain oikeanpuoleista piikkiä (mu2)
        peak_mask = (x_fit > mu2 - f * sigma2) & (x_fit < mu2 + f * sigma2)
    elif "1300um" in label:
        # Käytä vain vasemmanpuoleista piikkiä (mu1)
        peak_mask = (x_fit > mu1 - f * sigma1) & (x_fit < mu1 + f * sigma1)
    else:
        # Käytä molempia
        peak_mask1 = (x_fit > mu1 - f * sigma1) & (x_fit < mu1 + f * sigma1)
        peak_mask2 = (x_fit > mu2 - f * sigma2) & (x_fit < mu2 + f * sigma2)
        peak_mask = peak_mask1 | peak_mask2

    x_peak = x_fit[peak_mask]
    y_peak = y_fit[peak_mask]
    y_model_peak = model_func(x_peak)

    dof = len(x_peak) - 2
    if dof > 0:
        return calculate_reduced_chi2(x_peak, y_peak, y_model_peak, dof)
    else:
        return None




def update_plot():

    global _updating_link
    label = current_label
    auto_fit_A1 = auto_fit_params[label]["A1"]
    auto_fit_A2 = auto_fit_params[label]["A2"]
    auto_fit_sigma1 = auto_fit_params[label]["sigma1"]
    auto_fit_sigma2 = auto_fit_params[label]["sigma2"]
    auto_fit_mu1 = auto_fit_params[label]["mu1"]
    auto_fit_mu2 = auto_fit_params[label]["mu2"]

    slider_A1.set_active(not auto_fit_A1)
    slider_A2.set_active(not auto_fit_A2)
    slider_sigma1.set_active(not auto_fit_sigma1)
    slider_sigma2.set_active(not auto_fit_sigma2)
    slider_mu1.set_active(not auto_fit_mu1)
    slider_mu2.set_active(not auto_fit_mu2)

    ab_params[label] = (slider_a.val, slider_b.val)
    
    a, b = ab_params[label]

    result = fit_results.get(label)
    if result is None:
        ax.clear()
        ax.set_title(f"{label}: Fit failed or file not found")
        fig.canvas.draw_idle()
        return

    x_fit, y_fit, x_smooth, full_popt = result

    # --- HAE MUIDEN PARAMETRIEN KIINNITETYT ARVOT ---
    mu2_fixed = slider_mu2.val
    sigma1_fixed = slider_sigma1.val
    sigma2_fixed = slider_sigma2.val

    mu1_fixed = slider_mu1.val


    A1_manual, A2_manual = amp_params[label]
    A1, A2 = A1_manual, A2_manual

    sigma1_start = sigma1_fixed
    sigma2_start = sigma2_fixed
 
    sigma1_err, sigma2_err = get_sigma_uncertainties(
        label, default_sigma1=0.1, default_sigma2=0.1
    )

    
    sigma1_min = max(slider_sigma1.valmin, 0.9 * sigma1_start)
    sigma1_max = min(slider_sigma1.valmax, 1.1 * sigma1_start)
 
    sigma2_min = max(slider_sigma2.valmin, 0.9 * sigma2_start)
    sigma2_max = min(slider_sigma2.valmax, 1.1 * sigma2_start)

    if sigma1_min >= sigma1_max:
        sigma1_min = max(slider_sigma1.valmin, sigma1_start - 0.01)
        sigma1_max = min(slider_sigma1.valmax, sigma1_start + 0.01)

    if sigma2_min >= sigma2_max:
        sigma2_min = max(slider_sigma2.valmin, sigma2_start - 0.01)
        sigma2_max = min(slider_sigma2.valmax, sigma2_start + 0.01)

    free_names = []
    p0_free = []
    lower_free = []
    upper_free = []

    if auto_fit_A1:
        free_names.append("A1")
        p0_free.append(A1_manual)
        lower_free.append(0.0)
        upper_free.append(np.inf)

    if auto_fit_A2:
        free_names.append("A2")
        p0_free.append(A2_manual)
        lower_free.append(0.0)
        upper_free.append(np.inf)
        
        
    if auto_fit_mu1:
        free_names.append("mu1")
        p0_free.append(mu1_fixed)
        lower_free.append(slider_mu1.valmin)
        upper_free.append(slider_mu1.valmax)

    if auto_fit_mu2:
        free_names.append("mu2")
        p0_free.append(mu2_fixed)
        lower_free.append(slider_mu2.valmin)
        upper_free.append(slider_mu2.valmax)
        

    if auto_fit_sigma1:
        free_names.append("sigma1")
        p0_free.append(sigma1_start)
        lower_free.append(sigma1_min)
        upper_free.append(sigma1_max)

    if auto_fit_sigma2:
        free_names.append("sigma2")
        p0_free.append(sigma2_start)
        lower_free.append(sigma2_min)
        upper_free.append(sigma2_max)

    if free_names:
        def model_free(x, *pars):
            A1_local = A1_manual
            A2_local = A2_manual
            mu1_local = mu1_fixed
            mu2_local = mu2_fixed
            sigma1_local = sigma1_start
            sigma2_local = sigma2_start

            for name, val in zip(free_names, pars):
                if name == "A1":
                    A1_local = val
                elif name == "A2":
                    A2_local = val
                elif name == "mu1":
                    mu1_local = val
                elif name == "mu2":
                    mu2_local = val
                elif name == "sigma1":
                    sigma1_local = val
                elif name == "sigma2":
                    sigma2_local = val

            return double_gaussian_with_linear(
                x, A1_local, mu1_local, sigma1_local,
                A2_local, mu2_local, sigma2_local, a, b
            )

        try:
            popt_free, _ = curve_fit(
                model_free,
                x_fit,
                np.maximum(y_fit, 0),
                p0=p0_free,
                bounds=(lower_free, upper_free)
            )

            for name, val in zip(free_names, popt_free):
                if name == "A1":
                    A1 = val
                elif name == "A2":
                    A2 = val
                elif name == "mu1":
                    mu1_fixed = val
                elif name == "mu2":
                    mu2_fixed = val
                elif name == "sigma1":
                    sigma1_fixed = val
                elif name == "sigma2":
                    sigma2_fixed = val
        except Exception:
            pass


    if auto_fit_mu1 and abs(slider_mu1.val - mu1_fixed) > 1e-9:
        _updating_link = True
        slider_mu1.set_val(mu1_fixed)
        _updating_link = False

    if auto_fit_mu2 and abs(slider_mu2.val - mu2_fixed) > 1e-9:
        _updating_link = True
        slider_mu2.set_val(mu2_fixed)
        _updating_link = False






    
    

    # --- LASKETAAN KÄYRÄT ---
    g1 = A1 * np.exp(-(x_smooth - mu1_fixed)**2 / (2 * sigma1_fixed**2)) + a*x_smooth + b
    g2 = A2 * np.exp(-(x_smooth - mu2_fixed)**2 / (2 * sigma2_fixed**2)) + a*x_smooth + b
    bg = a*x_smooth + b
    total = g1 + g2 - bg

    # --- PIIRTO ---
    ax.clear()
    ax.step(x_fit, y_fit, where='mid', color='black')
    ax.plot(x_smooth, g1, color='blue')
    ax.plot(x_smooth, g2, color='green')
    ax.plot(x_smooth, bg, 'k--')
    ax.plot(x_smooth, total, 'r')

    # Tekstit
    fwhm = 2.355 * abs(sigma2_fixed)
    ax.text(0.05, 0.95, f"mu2 = {mu2_fixed:.2f} keV\nFWHM = {fwhm:.2f} keV",
            transform=ax.transAxes, va='top')

    # --- CHI2 ---
    def model_for_chi(x):
        return (
            A1 * np.exp(-(x - mu1_fixed)**2 / (2 * sigma1_fixed**2))
            + A2 * np.exp(-(x - mu2_fixed)**2 / (2 * sigma2_fixed**2))
            + a*x + b
        )

    reduced_chi2 = custom_chi2_segment(
        x_fit, y_fit,
        model_for_chi,
        label,
        mu1_fixed, sigma1_fixed,
        mu2_fixed, sigma2_fixed
    )


    

    if reduced_chi2 is not None:
        ax.text(0.05, 0.80, f"χ²_red = {reduced_chi2:.2f}", transform=ax.transAxes, va='top')
    else:
        ax.text(0.05, 0.80, "χ²_red = N/A", transform=ax.transAxes, va='top')

    fig.canvas.draw_idle()

radio.on_clicked(on_radio)


slider_b.on_changed(on_slider_b_change)
slider_a.on_changed(on_slider_a_change)
slider_mu1.on_changed(on_slider_mu1_change)
slider_mu2.on_changed(on_slider_mu2_change)
slider_sigma1.on_changed(lambda val: update_plot())
slider_sigma2.on_changed(lambda val: update_plot())



def update_slider_ranges(label):
    if "feeder" in label:
        new_range = (330, 429)
    else:
        new_range = (940, 980)

    # Päivitä raja-arvot
    slider_mu1.valmin, slider_mu1.valmax = new_range
    slider_mu2.valmin, slider_mu2.valmax = new_range
    slider_mu1.ax.set_xlim(new_range)
    slider_mu2.ax.set_xlim(new_range)

    # Pakota arvo kelvolliseksi – tämä on ratkaiseva:
    slider_mu1.set_val(np.clip(slider_mu1.val, *new_range))
    slider_mu2.set_val(np.clip(slider_mu2.val, *new_range))

    
     #   # Älä nollaa arvoja kun vaihdetaan labelia
      #  slider_sigma1.set_val(slider_sigma1.val)
       # slider_sigma2.set_val(slider_sigma2.val)
    #


ab_params = {label: (files[label][2][-2], files[label][2][-1]) for label in files}
amp_params = {label: (files[label][2][0], files[label][2][3]) for label in files}
on_radio(current_label)



from matplotlib.widgets import Button

# 1. Tallennussanakirja

# a ja b parametrien muisti



def calculate_area_and_error(A, sigma, dA, dsigma):
    """Laskee alueen ja sen virheen oikein."""
    area = A * sigma * np.sqrt(2 * np.pi)
    darea = np.sqrt(area)#/area
    return area, darea









def print_saved_table():
    if not saved_fits:
        print("🔍 Ei tallennettuja sovituksia vielä.")
        return

    print("\n📊 Tallennetut sovitusparametrit:")

    header = (
        f"{'Label':<18} {'A1':>6} {'mu1':>7} {'σ1':>6} {'A2':>6} {'mu2':>7} {'σ2':>6}"
        f"{'a':>6} {'b':>6} {'Area1':>8} {'Area2':>8} {'ratio1':>7} {'dRatio1':>8} {'ratio2':>7} {'dRatio2':>8}"
    )
    print(header)
    print("-" * len(header))

    ratios_only_file = "ratios_only_ring2_66As_8.txt"
    lines_dict = {}
    if os.path.exists(ratios_only_file):
        with open(ratios_only_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 1:
                    label = parts[0]
                    lines_dict[label] = line.strip()

    for label, params in saved_fits.items():
        A1, mu1, sigma1, A2, mu2, sigma2, a, b, Area1, Area2, sigma1_err, sigma2_err, dArea1, dArea2, dRatio1, dRatio2 = params

        # Hae tallennetut ratioarvot virheineen
        ratio1 = Area1 / (Area1 + Area2)
        ratio2 = Area2 / (Area1 + Area2)


        print(
            f"{label:<18} {A1:6.0f} {mu1:7.2f} {sigma1:6.2f} "
            f"{A2:6.0f} {mu2:7.2f} {sigma2:6.2f} {a:6.2f} {b:6.1f} "
            f"{Area1:8.1f} {Area2:8.1f} {ratio1:7.4f} {dRatio1:8.4f} {ratio2:7.4f} {dRatio2:8.4f}"
        )


        new_line = f"{label:<14} {ratio1:9.4f} {dRatio1:8.4f} {ratio2:9.4f} {dRatio2:8.4f}"
        lines_dict[label] = new_line

    with open(ratios_only_file, "w") as f:
        for line in lines_dict.values():
            f.write(line + "\n")



# 3. Tallennusnappi
button_ax = plt.axes([0.65, 0.05, 0.3, 0.05])
save_button = Button(button_ax, "Tallenna sovitusparametrit")

def save_current_fit(event=None):
    label = current_label
    auto_fit_A1 = auto_fit_params[label]["A1"]
    auto_fit_A2 = auto_fit_params[label]["A2"]
    auto_fit_sigma1 = auto_fit_params[label]["sigma1"]
    auto_fit_sigma2 = auto_fit_params[label]["sigma2"]
  
    result = fit_results.get(label)
    if result is None:
        print(f"❌ Ei sovitusta tallennettavaksi: {label}")
        return

    x_fit, y_fit, x_smooth, full_popt = result
    a, b = ab_params[label]


    if result is None:
        print(f"❌ Ei sovitusta tallennettavaksi: {label}")
        return

    if "decay" in label:
        ref_10 = fit_results.get("5um_decay")
        ref_290 = fit_results.get("1300um_decay")
        is_self_10 = label == "5um_decay"
        is_self_290 = label == "1300um_decay"
    elif "feeder" in label:
        ref_10 = fit_results.get("5um_feeder")
        ref_290 = fit_results.get("1300um_feeder")
        is_self_10 = label == "5um_feeder"
        is_self_290 = label == "1300um_feeder"
    else:
        print(f"❌ Tuntematon label: {label}")
        return




    if ref_10 is None or ref_290 is None:
        print(f"❌ Viitesovitukset puuttuvat ({label})")
        return
    
    # Tarkista onko tallennettava spektri itse viite
    is_self_10 = label == "5um_decay" if "decay" in label else label == "5um_feeder"
    is_self_290 = label == "1300um_decay" if "decay" in label else label == "1300um_feeder"
    
    _, _, _, popt_10 = ref_10
    _, _, _, popt_290 = ref_290
    _, _, _, popt_self = result
    
    # Päivitä mu1/mu2 ja sigma1/sigma2 oikeista lähteistä
    mu2_fixed = slider_mu2.val
    sigma1_fixed = slider_sigma1.val
    sigma2_fixed = slider_sigma2.val
    
    mu1_fixed = slider_mu1.val



    if label == "5um_decay":
        # Sovitetaan koko double_gaussian_with_linear uudestaan a:n ja b:n mukaan        
        
        
        try:
            popt_new, _ = curve_fit(
                double_gaussian_with_linear, x_fit, y_fit,
                p0=[full_popt[0], full_popt[1], full_popt[2],
                    full_popt[3], full_popt[4], full_popt[5], a, b],
                bounds=([0, 0, 1.0, 0, 0, 1.0, -np.inf, -np.inf],
                        [np.inf, np.inf, 8.0, np.inf, np.inf, 8.0, np.inf, np.inf])
            )
            # Päivitä popt_10
            fit_results["5um_decay"] = (x_fit, y_fit, x_smooth, popt_new)
            popt_10 = popt_new
        except:
            pass  # jätä ennalleen jos epäonnistuu
    elif label == "1300um_decay":
        try:
            popt_new, _ = curve_fit(
                double_gaussian_with_linear, x_fit, y_fit,
                p0=full_popt,
                bounds=([0, 0, 1.0, 0, 0, 1.0, -np.inf, -np.inf],
                        [np.inf, np.inf, 8.0, np.inf, np.inf, 8.0, np.inf, np.inf])
            )
            fit_results["1300um_decay"] = (x_fit, y_fit, x_smooth, popt_new)
            popt_290 = popt_new
        except:
            pass


    x_fit, y_fit, x_smooth, _ = result
    a, b = ab_params[label]


    # Sovita A1 ja A2 uudelleen kiinnitetyillä arvoilla
    def model_fixed(x, A1, A2):
        return double_gaussian_with_linear(x, A1, mu1_fixed, sigma1_fixed, A2, mu2_fixed, sigma2_fixed, a, b)

    try: 
        popt_partial, _ = curve_fit(model_fixed, x_fit, np.maximum(y_fit,0), p0=[1000, 1000])

    except:
        popt_partial = [0, 0]

#    A1, A2 = amp_params[label]
    
    A1_manual, A2_manual = amp_params[label]
    A1, A2 = A1_manual, A2_manual

    sigma1_start = sigma1_fixed
    sigma2_start = sigma2_fixed

    sigma1_err_saved, sigma2_err_saved = get_sigma_uncertainties(
        label, default_sigma1=0.1, default_sigma2=0.1
    )

    sigma1_min = max(slider_sigma1.valmin, 0.9 * sigma1_start)
    sigma1_max = min(slider_sigma1.valmax, 1.1 * sigma1_start)

    sigma2_min = max(slider_sigma2.valmin, 0.9 * sigma2_start)
    sigma2_max = min(slider_sigma2.valmax, 1.1 * sigma2_start)

    if sigma1_min >= sigma1_max:
        sigma1_min = max(slider_sigma1.valmin, sigma1_start - 0.01)
        sigma1_max = min(slider_sigma1.valmax, sigma1_start + 0.01)

    if sigma2_min >= sigma2_max:
        sigma2_min = max(slider_sigma2.valmin, sigma2_start - 0.01)
        sigma2_max = min(slider_sigma2.valmax, sigma2_start + 0.01)

    free_names = []
    p0_free = []
    lower_free = []
    upper_free = []

    if auto_fit_A1:
        free_names.append("A1")
        p0_free.append(A1_manual)
        lower_free.append(0.0)
        upper_free.append(np.inf)

    if auto_fit_A2:
        free_names.append("A2")
        p0_free.append(A2_manual)
        lower_free.append(0.0)
        upper_free.append(np.inf)

    if auto_fit_sigma1:
        free_names.append("sigma1")
        p0_free.append(sigma1_start)
        lower_free.append(sigma1_min)
        upper_free.append(sigma1_max)

    if auto_fit_sigma2:
        free_names.append("sigma2")
        p0_free.append(sigma2_start)
        lower_free.append(sigma2_min)
        upper_free.append(sigma2_max)

    if free_names:
        def model_free(x, *pars):
            A1_local = A1_manual
            A2_local = A2_manual
            sigma1_local = sigma1_start
            sigma2_local = sigma2_start

            for name, val in zip(free_names, pars):
                if name == "A1":
                    A1_local = val
                elif name == "A2":
                    A2_local = val
                elif name == "sigma1":
                    sigma1_local = val
                elif name == "sigma2":
                    sigma2_local = val

            return double_gaussian_with_linear(
                x, A1_local, mu1_fixed, sigma1_local,
                A2_local, mu2_fixed, sigma2_local, a, b
            )

        try:
            popt_free, _ = curve_fit(
                model_free,
                x_fit,
                np.maximum(y_fit, 0),
                p0=p0_free,
                bounds=(lower_free, upper_free)
            )

            for name, val in zip(free_names, popt_free):
                if name == "A1":
                    A1 = val
                elif name == "A2":
                    A2 = val
                elif name == "mu1":
                    mu1_fixed = val
                elif name == "mu2":
                    mu2_fixed = val
                elif name == "sigma1":
                    sigma1_fixed = val
                elif name == "sigma2":
                    sigma2_fixed = val
        except Exception:
            pass
   

    def model_amp(x, A1, A2):
        return double_gaussian_with_linear(x, A1, mu1_fixed, sigma1_fixed,
                                           A2, mu2_fixed, sigma2_fixed, a, b)

    if auto_fit_A1 and not auto_fit_A2:
        try:
            (A1_fit, _), _ = curve_fit(model_amp, x_fit, np.maximum(y_fit,0),
                                       p0=[A1_manual, A2_manual],
                                       bounds=([0,0],[np.inf,np.inf]))
            A1 = A1_fit
        except:
            pass

    elif auto_fit_A2 and not auto_fit_A1:
        try:
            (_, A2_fit), _ = curve_fit(model_amp, x_fit, np.maximum(y_fit,0),
                                       p0=[A1_manual, A2_manual],
                                       bounds=([0,0],[np.inf,np.inf]))
            A2 = A2_fit
        except:
            pass

    elif auto_fit_A1 and auto_fit_A2:
        try:
            (A1_fit, A2_fit), _ = curve_fit(model_amp, x_fit, np.maximum(y_fit,0),
                                            p0=[A1_manual, A2_manual],
                                            bounds=([0,0],[np.inf,np.inf]))
            A1, A2 = A1_fit, A2_fit
        except:
            pass












    else:
        # Käytä slider-arvoja
        A1, A2 = amp_params[label]





    final_popt = [A1, mu1_fixed, sigma1_fixed, A2, mu2_fixed, sigma2_fixed, a, b]
    
    # Aseta oletus epävarmuuksille ennen yritystä
    sigma1_err = 0.1
    sigma2_err = 0.1



    try:
        full_fit, full_cov = curve_fit(
 	   double_gaussian_with_linear, x_fit, np.maximum(y_fit,0),

            p0=final_popt,
            bounds=([0, 0, 1.0, 0, 0, 1.0, -np.inf, -np.inf],
                    [np.inf, np.inf, 8.0, np.inf, np.inf, 8.0, np.inf, np.inf])
        )

        mu1_err = np.sqrt(full_cov[1, 1])
        mu2_err = np.sqrt(full_cov[4, 4])
        sigma1_err = np.sqrt(full_cov[2, 2])
        sigma2_err = np.sqrt(full_cov[5, 5])

        
        # 🔧 Laske ensin piikkien alat


        
        dA1 = dA2 = 0.0
        Area1 = A1 * sigma1_fixed * np.sqrt(2*np.pi)
        Area2 = A2 * sigma2_fixed * np.sqrt(2*np.pi)
        dArea1 = np.sqrt(max(Area1, 1e-12))
        dArea2 = np.sqrt(max(Area2, 1e-12))
        ratio1 = Area1 / max(Area1 + Area2, 1e-12)
        ratio2 = Area2 / max(Area1 + Area2, 1e-12)
        dRatio1 = ratio1 * dArea1 / max(Area1, 1e-12)
        dRatio2 = ratio2 * dArea2 / max(Area2, 1e-12)

        total_area = Area1 + Area2

       # dRatio1 = ratio1 * np.sqrt((dArea1 / Area1)**2 + (dArea2 / Area2)**2)#11.8.2025
       # dRatio2 = ratio2 * np.sqrt((dArea1 / Area1)**2 + (dArea2 / Area2)**2)
  
        print("📈 Sovitusparametrien epävarmuudet:")
        print(f"A1     = {A1:.2f} ± {dA1:.2f}")
        print(f"A2     = {A2:.2f} ± {dA2:.2f}")
        print(f"σ1     = {sigma1_fixed:.3f} ± {sigma1_err:.3f}")
        print(f"σ2     = {sigma2_fixed:.3f} ± {sigma2_err:.3f}")
        print(f"mu1     = {mu1_fixed:.3f} ± {mu1_err:.3f}")
        print(f"mu2     = {mu2_fixed:.3f} ± {mu2_err:.3f}")
        print(f"Area1  = {Area1:.2f} ± {dArea1:.2f}")
        print(f"Area2  = {Area2:.2f} ± {dArea2:.2f}")
        print(f"ratio1 = {ratio1:.4f} ± {dRatio1:.4f}")
        print(f"ratio2 = {ratio2:.4f} ± {dRatio2:.4f}")
    
    

    
    except Exception as e:
        print(f"⚠️ Epävarmuuden arviointi epäonnistui: {e}")
        # Käytetään fallback-arvoja, jotka on jo asetettu (0.1)


    area1 = A1 * sigma1_fixed * np.sqrt(2 * np.pi)
    area2 = A2 * sigma2_fixed * np.sqrt(2 * np.pi)

    if label in saved_fits:
        del saved_fits[label]
    
    saved_fits[label] = np.round(
        final_popt + [Area1, Area2, sigma1_err, sigma2_err, dArea1, dArea2, dRatio1, dRatio2], 4
    )




    print(f"✅ Sovitus tallennettu: {label}")
    print_saved_table()

    # 🔽 Kirjoita tiedostoon
    decay_file = "params_decay_ring2_66As_8.txt"
    feeder_file = "params_feeder_ring2_66As_8.txt"

    out_line = f"{label:<18} {A1:.1f} {mu1_fixed:.3f} {sigma1_fixed:.3f} {A2:.1f} {mu2_fixed:.3f} {sigma2_fixed:.3f} {a:.4f} {b:.2f}\n"

    if "decay" in label:
        overwrite_entry_in_file(decay_file, label, out_line)
    elif "feeder" in label:
        overwrite_entry_in_file(feeder_file, label, out_line)


    write_int_ring2()


    import math


    if "decay" in label or "feeder" in label:
        Area1, dArea1, Area2, dArea2, ratio1, dRatio1, ratio2, dRatio2 = calculate_ratio_and_error(
            A1, sigma1_fixed, A2, sigma2_fixed, dA1, dA2, sigma1_err, sigma2_err
        )

        if "decay" in label:
            ratios_file = "gaussian_peak_areas_decay_with_ratios_ring2_66As_8.txt"
        else:
            ratios_file = "gaussian_peak_areas_feeder_with_ratios_ring2_66As_8.txt"

        line = (
            f"{label:<14} "
            f"{Area1:9.4f} {dArea1:8.4f} "
            f"{Area2:9.4f} {dArea2:8.4f} "
            f"{ratio1:9.4f} {dRatio1:8.4f} "
            f"{ratio2:9.4f} {dRatio2:8.4f}\n"
        )
        overwrite_entry_in_file(ratios_file, label, line)

        ratios_only_file = "ratios_only_ring2_66As_8.txt"
        ratios_only_line = (
            f"{label:<14} "
            f"{ratio1:9.4f} {dRatio1:8.4f} "
            f"{ratio2:9.4f} {dRatio2:8.4f}\n"
        )
        ratio_info = (ratio1, dRatio1, ratio2, dRatio2)
        overwrite_entry_in_file(ratios_only_file, label, ratios_only_line, ratio_info)



        # 🔽 Tallenna ratioarvot oikeilla epävarmuuksilla myös tähän tiedostoon
        ratios_only_file = "ratios_only_ring2_66As_8.txt"



    

# 5. Kytke nappi toimintaan
save_button.on_clicked(save_current_fit)


all_ax = plt.axes([0.32, 0.05, 0.3, 0.05])
all_button = Button(all_ax, "Näytä kaikki spektrit")
all_button.on_clicked(draw_all_spectra)


plt.show() 






import numpy as np
import pandas as pd

# Lue decay- ja feeder-tiedostot
decay_df = pd.read_csv("gaussian_peak_areas_decay_with_ratios_ring2_66As_8.txt", sep="\s+", header=None)
feeder_df = pd.read_csv("gaussian_peak_areas_feeder_with_ratios_ring2_66As_8.txt", sep="\s+", header=None)

# Nimeä sarakkeet
cols = [
    "label", "Area1", "dArea1", "Area2", "dArea2", "AreaRatio1", "dAreaRatio1",  "AreaRatio2", "dAreaRatio2"
]
decay_df.columns = feeder_df.columns = cols

# Ota etäisyys labelista
decay_df["d"] = decay_df["label"].str.extract(r"(\d+)").astype(float)
feeder_df["d"] = feeder_df["label"].str.extract(r"(\d+)").astype(float)

# Lajittele etäisyyden mukaan
decay_df = decay_df.sort_values("d")
feeder_df = feeder_df.sort_values("d")

# Laske suhteet ja virheet
def compute_fractions(df):
    Area1 = df["Area1"].values
    Area2 = df["Area2"].values
    dA1 = df["dArea1"].values
    dA2 = df["dArea2"].values
    total = Area1 + Area2

    I_S = Area1 / total
    I_D = Area2 / total

    dI_S = I_S * np.sqrt((dA1 / Area1)**2 + (dA2 / total)**2)
    dI_D = I_D * np.sqrt((dA2 / Area2)**2 + (dA1 / total)**2)

    return I_S, dI_S, I_D, dI_D

I_S_decay, sig_I_S_decay, I_D_decay, sig_I_D_decay = compute_fractions(decay_df)
I_S_feeder, sig_I_S_feeder, I_D_feeder, sig_I_D_feeder = compute_fractions(feeder_df)

# Kokoa lopullinen DataFrame
output_df = pd.DataFrame({
    "d": decay_df["d"].values / 1e6,  # µm -> m
    "I_S_decay": I_S_decay,
    "sig_I_S_decay": sig_I_S_decay,
    "I_D_decay": I_D_decay,
    "sig_I_D_decay": sig_I_D_decay,
    "I_S_feeder": I_S_feeder,
    "sig_I_S_feeder": sig_I_S_feeder,
    "I_D_feeder": I_D_feeder,
    "sig_I_D_feeder": sig_I_D_feeder,
})






# Lue syötetiedosto ja muodosta uusi tiedosto annetulla rakenteella

label_map = {
    "5um": 5,
    "270um": 270,
    "700um": 700,
    "1300um": 1300
}

last_column = {
    5: 0.1,
    270: 0.2,
    700: 0.3,
    1300: 0.4
}

data_decay = {}
data_feeder = {}

with open("ratios_only_ring2_66As_8.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        label = parts[0]
        values = list(map(float, parts[1:]))
        for prefix, key in label_map.items():
            if label.startswith(f"{prefix}_decay"):
                data_decay[key] = values
            elif label.startswith(f"{prefix}_feeder"):
                data_feeder[key] = values

with open("int_ring2_66As_8.fit", "w") as f_out:
    for key in sorted(label_map.values()):
        row = [key]
        row.extend(data_decay.get(key, [0.0]*4))
        row.extend(data_feeder.get(key, [0.0]*4))
     #   row.append(last_column.get(key, 0.0))
        f_out.write(" ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + "\n") 
