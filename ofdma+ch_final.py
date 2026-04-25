# @title
"""
NTN NB-IoT OFDMA Simulation 
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import itur
from scipy.special import erfc
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

np.random.seed(42)

ITUR_AVAILABLE = True
CHANNEL_GRAPHICS_AVAILABLE = False
OFDMA_GRAPHICS_AVAILABLE = True

def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def generate_bits(length):
    return np.random.binomial(n=1, p=0.5, size=length)

def ensure_even_bits(bits):
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    return bits

def symbol_gen(bin_data):
    return np.array([bin_data[i:i+2] for i in range(0, len(bin_data), 2)])

def map_to_constellation(symbols, A):
    constellation_map = {
        (0, 0): (-A, -A),
        (0, 1): (-A,  A),
        (1, 1): ( A,  A),
        (1, 0): ( A, -A)
    }
    return np.array([constellation_map[tuple(dibit)] for dibit in symbols])

def qpsk_modulate(bits, A):
    bits = ensure_even_bits(bits)
    dibits = symbol_gen(bits)
    pts = map_to_constellation(dibits, A)
    return pts[:, 0] + 1j * pts[:, 1]

def add_cp(ofdm_t, cp_length):
    return np.hstack([ofdm_t[-cp_length:], ofdm_t])

def remove_cp(signal, cp_length, sub):
    return signal[cp_length:cp_length + sub]

def nearest_neighbor_qpsk(symbol_rx, A):
    constellation_map = {
        (-A, -A): [0, 0],
        (-A,  A): [0, 1],
        ( A, -A): [1, 0],
        ( A,  A): [1, 1]
    }
    pts  = np.array(list(constellation_map.keys()))
    vals = np.array(list(constellation_map.values()))
    bits = []
    for sym in symbol_rx:
        diff = pts - np.array([sym.real, sym.imag])
        nearest = np.argmin(np.sum(diff**2, axis=1))
        bits.extend(vals[nearest])
    return np.array(bits)

def awgn(sig, snr_db, cp_len, fft_size, active_subcarriers):
    correction  = 10 * np.log10(active_subcarriers / fft_size)
    actual_snr  = snr_db + correction
    snr_linear  = 10 ** (actual_snr / 10)
    signal_power = np.mean(np.abs(sig) ** 2)
    noise_power  = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2)
    noise = sigma * (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape))
    return sig + noise

# ===========================================================================
# SECTION 1 – STATE-BASED SATELLITE CHANNEL  
# ===========================================================================
print("\n" + "="*60)
print("  SECTION 1 – State-based satellite channel model")
print("="*60)

def compute_channel_series():
    ts    = load.timescale()
    t_now = ts.now()

    tle_line1 = '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997'
    tle_line2 = '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000'

    satellite      = EarthSatellite(tle_line1, tle_line2, 'CONNECTA', ts)
    station_lat    = 39.9208
    station_lon    = 32.8541
    ground_station = wgs84.latlon(station_lat, station_lon)

    carrier_freq_mhz = 2000
    carrier_freq_ghz = carrier_freq_mhz / 1000.0
    carrier_freq_hz  = carrier_freq_mhz * 1e6
    c_km_s           = 299792.458

    tx_power_dbm      = 30.0
    tx_gain_dbi       = 20.0
    rx_gain_dbi       = 0.0
    rx_sensitivity_dbm = -125.0

    itu_exceedance_percent = 1.0
    earth_station_diameter_m = 0.2

    K_BOLT          = 1.380649e-23
    T_SYS           = 290.0
    BW_HZ           = 180e3
    noise_power_dbm = 10 * np.log10(K_BOLT * T_SYS * BW_HZ / 1e-3)

    t_future     = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))
    pass_times, pass_events = satellite.find_events(
        ground_station, t_now, t_future, altitude_degrees=5.0)
    rise_indices = [i for i, ev in enumerate(pass_events) if ev == 0]

    diff           = satellite - ground_station
    best_culm_time = None
    best_max_el    = -999.0

    for pi in range(len(rise_indices)):
        ri = rise_indices[pi]; ci = ri + 1; si = ri + 2
        if si >= len(pass_times): continue
        if pass_events[ci] != 1 or pass_events[si] != 2: continue
        alt, _, _ = diff.at(pass_times[ci]).altaz()
        if alt.degrees > best_max_el:
            best_max_el    = alt.degrees
            best_culm_time = pass_times[ci]

    if best_culm_time is None:
        if len(rise_indices) > 0:
            idx_r = rise_indices[0]
            best_culm_time = pass_times[idx_r + 1] if idx_r + 1 < len(pass_times) else pass_times[idx_r]
        else:
            best_culm_time = t_now

    WINDOW_HALF = timedelta(minutes=5)
    t_start     = ts.from_datetime(best_culm_time.utc_datetime() - WINDOW_HALF)

    TOTAL_SECONDS = 600

    time_minutes_list    = []
    elevation_list       = []
    distance_list        = []
    doppler_list         = []
    fspl_list            = []
    total_path_loss_list = []
    k_factor_list        = []
    link_budget_list     = []
    snr_list             = []
    sat_lat_list         = []
    sat_lon_list         = []
    markov_list          = []
    atm_gas_list         = []
    atm_cloud_list       = []
    atm_rain_list        = []
    atm_scint_list       = []
    atm_total_list       = []

    def get_itur_atmospheric_loss(lat, lon, freq_ghz, elevation_deg, p_percent, antenna_diameter_m):
        el_safe = max(float(elevation_deg), 1.0)
        Ag, Ac, Ar, As, At = itur.atmospheric_attenuation_slant_path(
            lat, lon, freq_ghz, el_safe, p_percent, antenna_diameter_m,
            return_contributions=True,
            include_rain=True,
            include_gas=True,
            include_scintillation=True,
            include_clouds=True
        )
        return float(Ag.value), float(Ac.value), float(Ar.value), float(As.value), float(At.value)

    for s in range(TOTAL_SECONDS):
            current_t = ts.from_datetime(t_start.utc_datetime() + timedelta(seconds=s))
            alt, az, dist_obj = diff.at(current_t).altaz()
            el_deg = alt.degrees
            d_km   = dist_obj.km

            time_minutes_list.append(s / 60.0)
            elevation_list.append(el_deg)
            distance_list.append(d_km)

            sp = wgs84.subpoint(satellite.at(current_t))
            sat_lat_list.append(sp.latitude.degrees)
            sat_lon_list.append(sp.longitude.degrees)

            t_next  = ts.from_datetime(current_t.utc_datetime() + timedelta(seconds=1))
            d_next  = diff.at(t_next).distance().km
            v_rel   = d_km - d_next
            doppler_hz = (v_rel / c_km_s) * carrier_freq_hz
            doppler_list.append(doppler_hz)

            el_s = max(el_deg, 1.0)
            fspl  = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(carrier_freq_mhz)
            atm_gas, atm_cloud, atm_rain, atm_scint, atm = get_itur_atmospheric_loss(
                station_lat, station_lon, carrier_freq_ghz, el_s,
                itu_exceedance_percent, earth_station_diameter_m
            )
            total_loss = fspl + atm
            fspl_list.append(fspl)
            total_path_loss_list.append(total_loss)
            atm_gas_list.append(atm_gas)
            atm_cloud_list.append(atm_cloud)
            atm_rain_list.append(atm_rain)
            atm_scint_list.append(atm_scint)
            atm_total_list.append(atm)

            k_db = min(15.0, 2.0 + ((el_s - 10) / 80) * 13.0)
            k_factor_list.append(k_db)

            rx_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss
            snr_db   = rx_power - noise_power_dbm
            link_budget_list.append(rx_power)
            snr_list.append(snr_db)

            if   el_deg >= 50.0: markov_list.append('A')
            elif el_deg >= 25.0: markov_list.append('B')
            else:                markov_list.append('C')

            if s % 30 == 0:
                print(f"  t={s/60:.2f}min | El={el_deg:.2f}° | d={d_km:.1f}km "
                    f"| Dop={doppler_hz:.1f}Hz | ITU-R Atm={atm:.2f}dB | SNR={snr_db:.2f}dB | State={markov_list[-1]}")

    return {
        'time_arr'          : np.array(time_minutes_list),
        'el_arr'            : np.array(elevation_list),
        'dist_arr'          : np.array(distance_list),
        'dop_arr'           : np.array(doppler_list),
        'fspl_arr'          : np.array(fspl_list),
        'loss_arr'          : np.array(total_path_loss_list),
        'k_arr'             : np.array(k_factor_list),
        'lb_arr'            : np.array(link_budget_list),
        'snr_arr'           : np.array(snr_list),
        'atm_gas_arr'       : np.array(atm_gas_list),
        'atm_cloud_arr'     : np.array(atm_cloud_list),
        'atm_rain_arr'      : np.array(atm_rain_list),
        'atm_scint_arr'     : np.array(atm_scint_list),
        'atm_total_arr'     : np.array(atm_total_list),
        'sat_lat_arr'       : np.array(sat_lat_list),
        'sat_lon_arr'       : np.array(sat_lon_list),
        'state_arr'         : np.array(markov_list),
        'rx_sensitivity_dbm': -125.0,
        'noise_power_dbm'   : noise_power_dbm,
    }

def resample_channel_to_symbols(channel_results, num_ofdm_symbols):
    valid_len = len(channel_results['time_arr'])
    idx = np.linspace(0, valid_len - 1, num_ofdm_symbols).astype(int)
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim == 1
                       and len(v) == valid_len else v)
            for k, v in channel_results.items()}

channel_results = compute_channel_series()

el_arr    = channel_results['el_arr']
time_arr  = channel_results['time_arr']
dist_arr  = channel_results['dist_arr']
dop_arr   = channel_results['dop_arr']
fspl_arr  = channel_results['fspl_arr']
loss_arr  = channel_results['loss_arr']
k_arr     = channel_results['k_arr']
lb_arr    = channel_results['lb_arr']
snr_arr   = channel_results['snr_arr']
atm_gas_arr = channel_results['atm_gas_arr']
atm_cloud_arr = channel_results['atm_cloud_arr']
atm_rain_arr = channel_results['atm_rain_arr']
atm_scint_arr = channel_results['atm_scint_arr']
atm_total_arr = channel_results['atm_total_arr']
sat_lat_arr = channel_results['sat_lat_arr']
sat_lon_arr = channel_results['sat_lon_arr']
state_arr   = channel_results['state_arr']
state_counts = {s: list(state_arr).count(s) for s in ['A', 'B', 'C']}

print(f"\n  Elevation  : min={el_arr.min():.2f}°  max={el_arr.max():.2f}°")
print(f"  Slant range: min={dist_arr.min():.2f} km  max={dist_arr.max():.2f} km")
print(f"  Doppler    : min={dop_arr.min():.2f} Hz  max={dop_arr.max():.2f} Hz")
print(f"  ITU-R atm  : min={atm_total_arr.min():.2f} dB  max={atm_total_arr.max():.2f} dB")
print(f"  SNR        : min={snr_arr.min():.2f} dB  max={snr_arr.max():.2f} dB")
print(f"  Markov     : LOS={state_counts['A']}s  Rician={state_counts['B']}s  Rayleigh/Shadowing={state_counts['C']}s")

STATE_BG    = {'A': '#c8e6c9', 'B': '#fff9c4', 'C': '#ffcdd2'}
STATE_LABEL = {'A': 'State A – LOS (el ≥ 50°)',
               'B': 'State B – Rician (25°–50°)',
               'C': 'State C – Rayleigh/Shadowing (< 25°)'}


def add_markov_bands(ax):
    prev = state_arr[0]; seg = time_arr[0]; drawn = set()
    for i in range(1, len(time_arr)):
        if state_arr[i] != prev or i == len(time_arr) - 1:
            end = time_arr[i] if i < len(time_arr) - 1 else time_arr[-1]
            lbl = STATE_LABEL[prev] if prev not in drawn else None
            ax.axvspan(seg, end, alpha=0.18, color=STATE_BG[prev], label=lbl, zorder=0)
            drawn.add(prev); seg = time_arr[i]; prev = state_arr[i]

def add_culm(ax):
    ax.axvline(5.0, color='gray', linestyle=':', linewidth=1.3,
               alpha=0.9, label='Max Elevation (t=5 min)')

def fmt(ax, title, xlabel='Time (Minutes)', ylabel=''):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

# ===========================================================================
# SECTION 2 – CHANNEL FIGURES 
# ===========================================================================

if CHANNEL_GRAPHICS_AVAILABLE:
    fig1, ax = plt.subplots(figsize=(10, 5))
    fig1.canvas.manager.set_window_title('Fig 1 – Total Path Loss')
    add_markov_bands(ax)
    ax.plot(time_arr, loss_arr, color='crimson', linewidth=2,
            label='Total Path Loss (FSPL + ITU-R Atmospheric)', zorder=3)
    add_culm(ax)
    fmt(ax, '1. Total Path Loss Over Time', ylabel='Loss (dB)')
    fig1.tight_layout()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig_atm, ax_atm = plt.subplots(figsize=(10, 5))
    fig_atm.canvas.manager.set_window_title('Fig – ITU-R Atmospheric Loss')
    add_markov_bands(ax_atm)
    ax_atm.plot(time_arr, atm_gas_arr, linewidth=1.5, label='Gas')
    ax_atm.plot(time_arr, atm_cloud_arr, linewidth=1.5, label='Cloud')
    ax_atm.plot(time_arr, atm_rain_arr, linewidth=1.5, label='Rain')
    ax_atm.plot(time_arr, atm_scint_arr, linewidth=1.5, label='Scintillation')
    ax_atm.plot(time_arr, atm_total_arr, linewidth=2.2, label='Total ITU-R Atmospheric')
    add_culm(ax_atm)
    fmt(ax_atm, 'ITU-R Atmospheric Attenuation Contributions', ylabel='Loss (dB)')
    ax_atm.legend(fontsize=9)
    fig_atm.tight_layout()
    plt.show()

    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.canvas.manager.set_window_title('Fig 2 – FSPL & Slant Range')
    add_markov_bands(ax)
    c1, c2 = 'darkorange', 'steelblue'
    ln1, = ax.plot(time_arr, fspl_arr, color=c1, linewidth=2, label='FSPL (dB)', zorder=3)
    ax.set_ylabel('FSPL (dB)', color=c1, fontsize=10); ax.tick_params(axis='y', labelcolor=c1)
    ax.grid(True, linestyle='--', alpha=0.5)
    axt = ax.twinx()
    ln2, = axt.plot(time_arr, dist_arr, color=c2, linewidth=2, linestyle='-.', label='Slant Range (km)', zorder=3)
    axt.set_ylabel('Slant Range (km)', color=c2, fontsize=10); axt.tick_params(axis='y', labelcolor=c2)
    add_culm(ax)
    fmt(ax, '2. FSPL and Slant Range Over Time')
    ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()], fontsize=9, loc='upper right')
    fig2.tight_layout()
    plt.show()

    fig3, ax = plt.subplots(figsize=(10, 5))
    fig3.canvas.manager.set_window_title('Fig 3 – Elevation & Markov States')
    add_markov_bands(ax)
    ax.plot(time_arr, el_arr, color='royalblue', linewidth=2, label='Elevation (°)', zorder=3)
    ax.axhline(50.0, color='green',  linestyle='--', linewidth=1.3, alpha=0.85, label='50° A/B')
    ax.axhline(25.0, color='orange', linestyle='--', linewidth=1.3, alpha=0.85, label='25° B/C')
    add_culm(ax)
    fmt(ax, '3. Elevation Angle and Markov State Boundaries', ylabel='Elevation (°)')
    ax.legend(fontsize=9)
    fig3.tight_layout()
    plt.show()

    fig4, ax = plt.subplots(figsize=(10, 5))
    fig4.canvas.manager.set_window_title('Fig 4 – Doppler Shift (S-Curve)')
    add_markov_bands(ax)
    ax.plot(time_arr, dop_arr/1e3, color='darkorchid', linewidth=2.5, label='Doppler Shift', zorder=3)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Doppler')
    ax.fill_between(time_arr, 0, dop_arr/1e3, where=(dop_arr>0),
                    color='royalblue', alpha=0.10, label='Approaching (+f)')
    ax.fill_between(time_arr, 0, dop_arr/1e3, where=(dop_arr<0),
                    color='tomato',    alpha=0.10, label='Receding (−f)')
    zi = np.argmin(np.abs(dop_arr))
    ax.annotate(f'≈0 Hz @ {time_arr[zi]:.2f} min',
                xy=(time_arr[zi], dop_arr[zi]/1e3),
                xytext=(time_arr[zi]+0.5, dop_arr.max()/1e3*0.35),
                arrowprops=dict(arrowstyle='->', color='darkorchid', lw=1.6),
                fontsize=9, color='darkorchid', fontweight='bold')
    add_culm(ax)
    fmt(ax, '4. Doppler Shift (S-Curve)', ylabel='Doppler Frequency (kHz)')
    ax.legend(fontsize=9)
    fig4.tight_layout()
    plt.show()

    fig5, ax = plt.subplots(figsize=(10, 5))
    fig5.canvas.manager.set_window_title('Fig 5 – Rician K-Factor')
    add_markov_bands(ax)
    ax.plot(time_arr, k_arr, color='saddlebrown', linewidth=2, label='Rician K-Factor (dB)', zorder=3)
    add_culm(ax)
    fmt(ax, '5. Rician K-Factor Over Time', ylabel='K (dB)')
    ax.legend(fontsize=9)
    fig5.tight_layout()
    plt.show()

    rx_sens = channel_results['rx_sensitivity_dbm']
    fig6, ax = plt.subplots(figsize=(10, 5))
    fig6.canvas.manager.set_window_title('Fig 6 – Link Budget & SNR')
    add_markov_bands(ax)
    ax.plot(time_arr, lb_arr,  color='teal',  linewidth=2,   label='Rx Power (dBm)', zorder=3)
    ax.plot(time_arr, snr_arr, color='navy',  linewidth=1.5, linestyle='--', label='SNR (dB)', zorder=3)
    ax.axhline(rx_sens, color='red', linestyle='--', linewidth=2,
            label=f'Rx Sensitivity ({rx_sens} dBm)')
    ax.fill_between(time_arr, lb_arr.min()-5, rx_sens,
                    color='red',   alpha=0.08, label='Outage Zone',    zorder=1)
    ax.fill_between(time_arr, rx_sens, snr_arr.max()+5,
                    color='green', alpha=0.08, label='Link Available', zorder=1)
    add_culm(ax)
    fmt(ax, '6. Link Budget and SNR', ylabel='Power / SNR  (dBm / dB)')
    ax.set_ylim(lb_arr.min()-5, snr_arr.max()+5)
    ax.legend(fontsize=9, loc='lower right')
    fig6.tight_layout()
    plt.show()

    fig7, ax = plt.subplots(figsize=(9, 6))
    fig7.canvas.manager.set_window_title('Fig 7 – Markov State Parameters')
    states_lbl = ['A  (LOS)', 'B  (Moderate)', 'C  (Shadow)']
    el_bounds  = ['el ≥ 50°', '25°≤el<50°', 'el < 25°']
    durations  = [state_counts.get(k, 0) for k in ['A','B','C']]
    pct_       = [100*d/600 for d in durations]
    def state_mean_(arr, sk):
        idx_ = np.where(state_arr == sk)[0]
        return np.mean(arr[idx_]) if len(idx_) else float('nan')
    ml_ = [state_mean_(loss_arr, k) for k in ['A','B','C']]
    ms_ = [state_mean_(snr_arr,  k) for k in ['A','B','C']]
    mk_ = [state_mean_(k_arr,    k) for k in ['A','B','C']]
    bc_ = ['#2e7d32','#f9a825','#c62828']
    bars = ax.bar(states_lbl, durations, color=bc_, alpha=0.88, edgecolor='black', linewidth=0.9)
    for i,(bar,d,p,ml,ms,mk,bnd) in enumerate(zip(bars,durations,pct_,ml_,ms_,mk_,el_bounds)):
        if d>0:
            ax.text(bar.get_x()+bar.get_width()/2, max(d*0.06,4),
                    bnd, ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f'{d}s ({p:.0f}%)\nAvg Loss:{ml:.1f}dB\nAvg SNR:{ms:.1f}dB\nAvg K:{mk:.1f}dB',
                ha='center', va='bottom', fontsize=8.5)
    ax.set_ylim(0, max(durations)*1.60 if max(durations)>0 else 60)
    ax.set_ylabel('Duration (seconds)', fontsize=10)
    ax.set_xlabel('Channel State  (ITU-R / Lutz)', fontsize=10)
    ax.set_title('7. Markov State Parameters  —  10-min Window',
                fontsize=12, fontweight='bold', pad=8)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig7.tight_layout()
    plt.show()

    fig8 = plt.figure(figsize=(16, 7))
    fig8.canvas.manager.set_window_title('Fig 8 – Ground Track (Global)')
    ax8  = fig8.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    ax8.set_global()
    ax8.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
    ax8.add_feature(cfeature.LAND,      facecolor='#eaecee', zorder=0)
    ax8.add_feature(cfeature.COASTLINE, linewidth=0.8,        zorder=1)
    ax8.add_feature(cfeature.BORDERS,   linewidth=0.5, alpha=0.6, zorder=1)
    gl8 = ax8.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl8.top_labels=False; gl8.right_labels=False
    SC = {'A':'limegreen','B':'gold','C':'tomato'}
    for i in range(len(sat_lat_arr)-1):
        ax8.plot(sat_lon_arr[i:i+2], sat_lat_arr[i:i+2],
                color=SC[state_arr[i]], linewidth=2.8,
                transform=ccrs.PlateCarree(), zorder=3)
    if len(sat_lat_arr)>2:
        for ai in np.linspace(0, len(sat_lat_arr)-2, 6, dtype=int):
            ax8.annotate('', xy=(sat_lon_arr[ai+1],sat_lat_arr[ai+1]),
                        xytext=(sat_lon_arr[ai],sat_lat_arr[ai]),
                        arrowprops=dict(arrowstyle='->',color='navy',lw=1.5),
                        transform=ccrs.PlateCarree(), zorder=5)
    ax8.scatter(sat_lon_arr[0],  sat_lat_arr[0],  color='lime',  s=80, zorder=5, transform=ccrs.PlateCarree())
    ax8.scatter(sat_lon_arr[-1], sat_lat_arr[-1], color='black', s=80, zorder=5, transform=ccrs.PlateCarree())
    cx,cy = sat_lon_arr[300], sat_lat_arr[300]
    ax8.scatter(cx,cy, color='yellow', edgecolors='darkorange', linewidths=1.2,
                s=100, marker='*', zorder=6, transform=ccrs.PlateCarree())
    ax8.annotate('Max El.',(cx,cy), textcoords='offset points', xytext=(-9,-9),
                fontsize=9, color='darkorange', fontweight='bold', transform=ccrs.PlateCarree())
    ax8.scatter(32.8541, 39.9208, color='red', marker='^', s=100,
                zorder=6, transform=ccrs.PlateCarree())
    ax8.annotate('Ground Station',(32.8541, 39.9208),
                textcoords='offset points', xytext=(10,10),
                fontsize=9, color='darkred', fontweight='bold', transform=ccrs.PlateCarree())
    ax8.set_title('8. Satellite Ground Track  (green=State A / gold=State B / red=State C)',
                fontsize=11, fontweight='bold', pad=8)
    fig8.tight_layout()
    plt.show()

# ===========================================================================
# SECTION 3 – OFDMA PARAMETERS
# ===========================================================================
print("\n" + "="*60)
print("  SECTION 3 – OFDMA parameters")
print("="*60)

sub                    = 64
cp_length              = 16
system_active_subcarriers = 32
nb_iot_subcarriers     = 12
num_ofdm_symbols       = 10000
snr_bin_count          = 15

E_bit      = 1
A          = np.sqrt(E_bit) / np.sqrt(2)   # = 1/sqrt(2) = 0.7071
pilot_value = A + 1j * A

center     = sub // 2
half_nb    = nb_iot_subcarriers // 2
nb_iot_shifted = np.r_[np.arange(center - half_nb, center),
                        np.arange(center + 1, center + 1 + half_nb)]
nb_idx     = np.arange(nb_iot_subcarriers)

fs         = sub * 15000       # 960 kHz sampling frequency
block_len  = sub + cp_length   # 80 samples/block

def get_nrs_positions_in_symbol(symbol_in_subframe, n_cell_id):
    #v_shift = n_cell_id % 3
    #return np.array([v_shift, v_shift + 3, v_shift + 7, v_shift + 10])
    return np.array([0, 3, 8, 11])
"""
def get_data_positions_in_symbol(symbol_in_subframe, n_cell_id):
    nrs_local  = get_nrs_positions_in_symbol(symbol_in_subframe, n_cell_id)
    data_local = np.setdiff1d(np.arange(nb_iot_subcarriers), nrs_local)
    #half_data  = len(data_local) // 2
    #return nrs_local, data_local[:half_data], data_local[half_data:]
    ue1_local = data_local[0::2]  
    ue2_local = data_local[1::2]   
    return nrs_local, ue1_local, ue2_local
"""
def get_data_positions_in_symbol(symbol_in_subframe, n_cell_id):
    nrs_local  = get_nrs_positions_in_symbol(symbol_in_subframe, n_cell_id)
    data_local = np.setdiff1d(np.arange(nb_iot_subcarriers), nrs_local)
    
    # Ardışık değil, interleaved → her iki UE de tüm banda yayılır
    ue1_local = data_local[0::2]   # çift indeksler
    ue2_local = data_local[1::2]   # tek indeksler
    
    return nrs_local, ue1_local, ue2_local
print(f"  FFT={sub}  CP={cp_length}  NB={nb_iot_subcarriers}  fs={fs/1e3:.0f} kHz")
print(f"  NB-IoT shifted bins: {nb_iot_shifted}")
print("  Pilot optimization: 4 distributed pilots per OFDM symbol, 4 QPSK data symbols per UE")

# ===========================================================================
# SECTION 4 – RESOURCE GRID + TX  
# ===========================================================================
print("\n" + "="*60)
print("  SECTION 4 – Resource grid & TX")
print("="*60)

resource_map = np.zeros((num_ofdm_symbols, nb_iot_subcarriers), dtype=int)
for s in range(num_ofdm_symbols):
    nrs_l, ue1_l, ue2_l = get_data_positions_in_symbol(s % 14, n_cell_id=0)
    resource_map[s, ue1_l] = 2
    resource_map[s, ue2_l] = 3
    resource_map[s, nrs_l] = 1

if OFDMA_GRAPHICS_AVAILABLE:
    fig_rg, ax_rg = plt.subplots(figsize=(10, 4))
    fig_rg.canvas.manager.set_window_title('Fig – Resource Grid')
    ax_rg.imshow(resource_map.T, aspect='auto', origin='lower')
    ax_rg.set_yticks(np.arange(nb_iot_subcarriers))
    ax_rg.set_xlabel("OFDM Symbol Index")
    ax_rg.set_ylabel("NB-IoT Local Subcarrier Index")
    ax_rg.set_title("NB-IoT Downlink Resource Allocation Grid\n0=Empty, 1=Pilot, 2=UE1, 3=UE2")
    plt.colorbar(ax_rg.images[0], ax=ax_rg)
    plt.tight_layout()
    plt.show()

bits_per_symbol_ue1      = 2 * 4
bits_per_symbol_ue2      = 2 * 4
symbols_per_user_per_sym = 4

bits_tx_ue1   = generate_bits(num_ofdm_symbols * bits_per_symbol_ue1)
bits_tx_ue2   = generate_bits(num_ofdm_symbols * bits_per_symbol_ue2)
symbols_tx_ue1 = qpsk_modulate(bits_tx_ue1, A).reshape(num_ofdm_symbols, symbols_per_user_per_sym)
symbols_tx_ue2 = qpsk_modulate(bits_tx_ue2, A).reshape(num_ofdm_symbols, symbols_per_user_per_sym)

if OFDMA_GRAPHICS_AVAILABLE:
    fig_tx_c, ax_tc = plt.subplots(figsize=(6, 6))
    fig_tx_c.canvas.manager.set_window_title('Fig – TX Constellation')
    ax_tc.plot(np.real(symbols_tx_ue1.flatten()), np.imag(symbols_tx_ue1.flatten()),
            'bo', alpha=0.5, label='UE1 TX')
    ax_tc.plot(np.real(symbols_tx_ue2.flatten()), np.imag(symbols_tx_ue2.flatten()),
            'ro', alpha=0.5, label='UE2 TX')
    ax_tc.set_xlabel("Inphase"); ax_tc.set_ylabel("Quadrature")
    ax_tc.set_title("TX QPSK Constellations")
    ax_tc.grid(True); ax_tc.axis("equal"); ax_tc.legend()
    plt.tight_layout(); plt.show()

tx_blocks          = []
full_grid_shifted  = np.zeros((num_ofdm_symbols, sub), dtype=complex)
nb_grid            = np.zeros((num_ofdm_symbols, nb_iot_subcarriers), dtype=complex)

for s in range(num_ofdm_symbols):
    symbol_shifted = np.zeros(sub, dtype=complex)
    nrs_l, ue1_l, ue2_l = get_data_positions_in_symbol(s % 14, n_cell_id=0)
    nrs_sh = nb_iot_shifted[nrs_l]
    ue1_sh = nb_iot_shifted[ue1_l]
    ue2_sh = nb_iot_shifted[ue2_l]

    symbol_shifted[nrs_sh] = pilot_value
    symbol_shifted[ue1_sh] = symbols_tx_ue1[s]
    symbol_shifted[ue2_sh] = symbols_tx_ue2[s]

    full_grid_shifted[s]  = symbol_shifted
    nb_grid[s, nrs_l]  = pilot_value
    nb_grid[s, ue1_l]  = symbols_tx_ue1[s]
    nb_grid[s, ue2_l]  = symbols_tx_ue2[s]

    ofdm_t    = np.fft.ifft(np.fft.ifftshift(symbol_shifted)) * np.sqrt(sub)
    ofdm_t_cp = add_cp(ofdm_t, cp_length)
    tx_blocks.append(ofdm_t_cp)

tx_signal = np.concatenate(tx_blocks)

if OFDMA_GRAPHICS_AVAILABLE:
    fig_cm, ax_cm = plt.subplots(figsize=(10, 4))
    fig_cm.canvas.manager.set_window_title('Fig – 64-Bin Carrier Map')
    ax_cm.stem(np.arange(sub), np.abs(full_grid_shifted[0]), basefmt=" ")
    ax_cm.set_xlabel("Shifted Frequency Bin"); ax_cm.set_ylabel("Magnitude")
    ax_cm.set_title("Full 64-Bin Carrier Map for One OFDM Symbol")
    ax_cm.grid(True); plt.tight_layout(); plt.show()

win_tx   = np.hanning(len(tx_signal))
spec_tx  = np.fft.fftshift(np.fft.fft(tx_signal * win_tx, 4096))

if OFDMA_GRAPHICS_AVAILABLE:
    fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
    fig_sp.canvas.manager.set_window_title('Fig – TX Spectrum')
    ax_sp.plot(20 * np.log10(np.abs(spec_tx) / np.max(np.abs(spec_tx)) + 1e-12))
    ax_sp.set_xlabel("Frequency"); ax_sp.set_ylabel("Magnitude (dB)")
    ax_sp.set_title("OFDMA Spectrum with Guard Bands and NB-IoT Narrowband")
    ax_sp.grid(True); plt.tight_layout(); plt.show()

print(f"  TX signal: {len(tx_signal)} ornek")

# ===========================================================================
# SECTION 5 – CHANNEL EFFECT ON SIGNAL  (slot-based receiver)
# ===========================================================================
print("\n" + "="*60)
print("  SECTION 5 – Channel + Receiver  (slot-based freq-domain EQ)")
print("="*60)

channel_symbol     = resample_channel_to_symbols(channel_results, num_ofdm_symbols)
calculated_snr_arr = channel_symbol['snr_arr']
calculated_el_arr  = channel_symbol['el_arr']

rx_blocks_after_channel = []
h_channel_series        = []

def interp_channel(nrs_local, H_pilots, n_sub=12):
    f = interp1d(
        nrs_local.astype(float), H_pilots,
        kind='linear',
        bounds_error=False,
        fill_value=(H_pilots[0], H_pilots[-1])  
    )
    return f(np.arange(n_sub))

for s in range(num_ofdm_symbols):
    tx_block    = tx_blocks[s]
    doppler_hz  = channel_symbol['dop_arr'][s]
    k_factor_db = channel_symbol['k_arr'][s]
    snr_db      = calculated_snr_arr[s]
    t_block     = np.arange(len(tx_block)) / fs

    k_lin     = 10 ** (k_factor_db / 10)
    los       = np.sqrt(k_lin / (k_lin + 1))
    nlos      = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2 * (k_lin + 1))
    h_channel = los + nlos
    h_channel = h_channel / np.abs(h_channel)  

    ch_block         = tx_block * np.exp(1j * 2 * np.pi * doppler_hz * t_block) * h_channel
    rx_block_channel = awgn(ch_block, snr_db, cp_length, sub, nb_iot_subcarriers)

    rx_blocks_after_channel.append(rx_block_channel)
    h_channel_series.append(h_channel)

rx_signal = np.concatenate(rx_blocks_after_channel)

window_rx_before   = np.hanning(len(rx_signal))
spectrum_rx_before = np.fft.fftshift(np.fft.fft(rx_signal * window_rx_before, 4096))

rx_blocks_comp = []
phase_acc      = 0.0                         
idx            = 0

for s in range(num_ofdm_symbols):
    rx_block       = rx_signal[idx: idx + block_len]
    doppler_hz     = channel_symbol['dop_arr'][s]
    t_rx           = np.arange(len(rx_block)) / fs
    phase_vec      = phase_acc + 2 * np.pi * doppler_hz * t_rx
    rx_block_comp  = rx_block * np.exp(-1j * phase_vec)
    phase_acc      = phase_vec[-1] + 2 * np.pi * doppler_hz / fs  
    rx_blocks_comp.append(rx_block_comp)
    idx           += block_len

rx_signal_comp    = np.concatenate(rx_blocks_comp)
window_rx_after   = np.hanning(len(rx_signal_comp))
spectrum_rx_after = np.fft.fftshift(np.fft.fft(rx_signal_comp * window_rx_after, 4096))

rx_symbols_ue1 = np.zeros((num_ofdm_symbols, symbols_per_user_per_sym), dtype=complex)
rx_symbols_ue2 = np.zeros((num_ofdm_symbols, symbols_per_user_per_sym), dtype=complex)

raw_const_ue1 = []
raw_const_ue2 = []
eq_const_ue1  = []
eq_const_ue2  = []

for s in range(num_ofdm_symbols):

    blk       = rx_blocks_comp[s]
    ofdm_data = blk[cp_length: cp_length + sub]
    X_hat     = np.fft.fftshift(np.fft.fft(ofdm_data, n=sub)) / np.sqrt(sub)

    D_hat = np.array([X_hat[nb_iot_shifted[k]] for k in range(nb_iot_subcarriers)])

    nrs_local, ue1_local, ue2_local = get_data_positions_in_symbol(s % 14, n_cell_id=0)

    H_pilots = D_hat[nrs_local] / pilot_value   

    H_est_full = np.interp(
        np.arange(nb_iot_subcarriers),         
        nrs_local.astype(float),                
        H_pilots                                
    )
 
    H_est_full = (
        interp_channel(nrs_local, H_pilots.real)
        + 1j *
        interp_channel(nrs_local, H_pilots.imag)
    )

    H_est_ue1 = H_est_full[ue1_local]   
    H_est_ue2 = H_est_full[ue2_local]   

    raw_ue1 = D_hat[ue1_local]
    raw_ue2 = D_hat[ue2_local]

    raw_const_ue1.extend(raw_ue1)
    raw_const_ue2.extend(raw_ue2)

    snr_lin  = 10 ** (calculated_snr_arr[s] / 10)
    mmse_eps = 1.0 / (snr_lin + 1e-12)

    eq_ue1 = raw_ue1 * np.conj(H_est_ue1) / (np.abs(H_est_ue1)**2 + mmse_eps + 1e-12)
    eq_ue2 = raw_ue2 * np.conj(H_est_ue2) / (np.abs(H_est_ue2)**2 + mmse_eps + 1e-12)

    rx_symbols_ue1[s] = eq_ue1
    rx_symbols_ue2[s] = eq_ue2

    eq_const_ue1.extend(eq_ue1)
    eq_const_ue2.extend(eq_ue2)

bits_rx_ue1 = nearest_neighbor_qpsk(rx_symbols_ue1.flatten(), A)[:len(bits_tx_ue1)]
bits_rx_ue2 = nearest_neighbor_qpsk(rx_symbols_ue2.flatten(), A)[:len(bits_tx_ue2)]

ber_ue1_total = np.mean(bits_rx_ue1 != bits_tx_ue1)
ber_ue2_total = np.mean(bits_rx_ue2 != bits_tx_ue2)

instant_ber_ue1 = np.zeros(num_ofdm_symbols)
instant_ber_ue2 = np.zeros(num_ofdm_symbols)

for s in range(num_ofdm_symbols):
    b0 = s * bits_per_symbol_ue1
    b1 = (s + 1) * bits_per_symbol_ue1
    c0 = s * bits_per_symbol_ue2
    c1 = (s + 1) * bits_per_symbol_ue2
    rx_b_ue1_s = nearest_neighbor_qpsk(rx_symbols_ue1[s], A)[:bits_per_symbol_ue1]
    rx_b_ue2_s = nearest_neighbor_qpsk(rx_symbols_ue2[s], A)[:bits_per_symbol_ue2]
    instant_ber_ue1[s] = np.mean(rx_b_ue1_s != bits_tx_ue1[b0:b1])
    instant_ber_ue2[s] = np.mean(rx_b_ue2_s != bits_tx_ue2[c0:c1])

snr_min = np.floor(np.min(calculated_snr_arr))
snr_max = np.ceil(np.max(calculated_snr_arr))
snr_bins = np.linspace(snr_min, snr_max, snr_bin_count + 1)
snr_bin_centers = []
ber_bin_ue1     = []
ber_bin_ue2     = []
el_bin_mean     = []
sym_bin_count   = []

for i in range(len(snr_bins) - 1):
    if i == len(snr_bins) - 2:
        mask = (calculated_snr_arr >= snr_bins[i]) & (calculated_snr_arr <= snr_bins[i + 1])
    else:
        mask = (calculated_snr_arr >= snr_bins[i]) & (calculated_snr_arr < snr_bins[i + 1])
    if np.sum(mask) == 0:
        continue
    snr_bin_centers.append(np.mean(calculated_snr_arr[mask]))
    ber_bin_ue1.append(np.mean(instant_ber_ue1[mask]))
    ber_bin_ue2.append(np.mean(instant_ber_ue2[mask]))
    el_bin_mean.append(np.mean(calculated_el_arr[mask]))
    sym_bin_count.append(np.sum(mask))

snr_bin_centers = np.array(snr_bin_centers)
ber_bin_ue1     = np.array(ber_bin_ue1)
ber_bin_ue2     = np.array(ber_bin_ue2)
el_bin_mean     = np.array(el_bin_mean)
sym_bin_count   = np.array(sym_bin_count)
MIN_SYMS = 200
mask_valid      = sym_bin_count >= MIN_SYMS
snr_bin_centers = snr_bin_centers[mask_valid]
ber_bin_ue1     = ber_bin_ue1[mask_valid]
ber_bin_ue2     = ber_bin_ue2[mask_valid]
el_bin_mean     = el_bin_mean[mask_valid]
sym_bin_count   = sym_bin_count[mask_valid]

print(f"  Calculated SNR range from satellite pass: {np.min(calculated_snr_arr):.2f} dB to {np.max(calculated_snr_arr):.2f} dB")
print(f"  Overall UE1 BER = {ber_ue1_total:.8f}")
print(f"  Overall UE2 BER = {ber_ue2_total:.8f}")

if OFDMA_GRAPHICS_AVAILABLE:
    fig_cb, ax_cb = plt.subplots(figsize=(6, 6))
    fig_cb.canvas.manager.set_window_title('Fig – Constellation Before EQ')
    ax_cb.plot(np.real(raw_const_ue1), np.imag(raw_const_ue1),
            'bo', alpha=0.35, label='UE1 Before EQ')
    ax_cb.plot(np.real(raw_const_ue2), np.imag(raw_const_ue2),
            'ro', alpha=0.35, label='UE2 Before EQ')
    ax_cb.set_xlabel("Inphase"); ax_cb.set_ylabel("Quadrature")
    ax_cb.set_title("Received Constellation Before Equalization")
    ax_cb.grid(True); ax_cb.axis("equal"); ax_cb.legend()
    plt.tight_layout(); plt.show()

    fig14, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    fig14.canvas.manager.set_window_title('Fig 14 – Constellation After Equalisation')
    ax.plot(np.array(eq_const_ue1).real, np.array(eq_const_ue1).imag,
            'bo', alpha=0.25, markersize=3, label='UE1 after EQ')
    ax.plot(np.array(eq_const_ue2).real, np.array(eq_const_ue2).imag,
            'ro', alpha=0.25, markersize=3, label='UE2 after EQ')
    # Referans QPSK noktaları
    for xr, yi in [(-A,-A),(-A,A),(A,-A),(A,A)]:
        ax.plot(xr, yi, 'k+', markersize=14, markeredgewidth=2)
    ax.set_xlabel('In-phase'); ax.set_ylabel('Quadrature')
    ax.set_title('14. Received Constellation After Equalisation\n'
                 f'(QPSK ideal: ±{A:.4f})',
                 fontsize=11, fontweight='bold', pad=8)
    ax.grid(True); ax.axis('equal'); ax.legend(fontsize=9)
    fig14.tight_layout()
    plt.show()

    fig_spec, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(10 * np.log10(np.abs(spectrum_rx_before)**2), color='red')
    ax1.set_title("RX Spectrum: BEFORE Doppler Compensation")
    ax1.grid(True)
    ax2.plot(10 * np.log10(np.abs(spectrum_rx_after)**2), color='green')
    ax2.set_title("RX Spectrum: AFTER Doppler Compensation")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    fig_snr_time, ax_snr_time = plt.subplots(figsize=(10, 4))
    fig_snr_time.canvas.manager.set_window_title('Fig – Calculated SNR During Pass')
    ax_snr_time.plot(np.arange(num_ofdm_symbols), calculated_snr_arr, linewidth=1.8, label='Calculated SNR')
    ax_snr_time.set_xlabel("OFDM Symbol Index")
    ax_snr_time.set_ylabel("SNR (dB)")
    ax_snr_time.set_title("Calculated SNR During Satellite Pass")
    ax_snr_time.grid(True, linestyle='--', alpha=0.6)
    ax_snr_time.legend()
    plt.tight_layout()
    plt.show()

# ===========================================================================
# SECTION 6 – TERMINAL BER LIST
# ===========================================================================
print("\n" + "="*60)
print("  SECTION 6 – BER List (calculated satellite-pass SNR)")
print("="*60)

print()
print(f"  {'SNR(dB)':>10}  {'Mean El(deg)':>12}  {'Symbols':>8}  {'UE1 BER':>12}  {'UE2 BER':>12}")
print("  " + "-"*64)
for i in range(len(snr_bin_centers)):
    print(f"  {snr_bin_centers[i]:>10.2f}  {el_bin_mean[i]:>12.2f}  {sym_bin_count[i]:>8d}  {ber_bin_ue1[i]:>12.8f}  {ber_bin_ue2[i]:>12.8f}")

print()
print(f"  Overall UE1 BER={ber_ue1_total:.8f}")
print(f"  Overall UE2 BER={ber_ue2_total:.8f}")
print(f"  Average calculated SNR={np.mean(calculated_snr_arr):.2f} dB")
print(f"  Minimum calculated SNR={np.min(calculated_snr_arr):.2f} dB")
print(f"  Maximum calculated SNR={np.max(calculated_snr_arr):.2f} dB")

# ===========================================================================
# SECTION 7 – BER CURVE
# ===========================================================================

if OFDMA_GRAPHICS_AVAILABLE:
    fig_ber, ax_ber = plt.subplots(figsize=(8, 5))
    fig_ber.canvas.manager.set_window_title('Fig – BER Curve')
    ax_ber.semilogy(snr_bin_centers, ber_bin_ue1 + 1e-12, 'bo-',
                    label='UE1 Simulated BER', linewidth=1.5, markersize=6)
    ax_ber.semilogy(snr_bin_centers, ber_bin_ue2 + 1e-12, 'ro-',
                    label='UE2 Simulated BER', linewidth=1.5, markersize=6)
    ax_ber.set_xlabel("Calculated SNR from Satellite Pass (dB)")
    ax_ber.set_ylabel("BER")
    ax_ber.set_title("NB-IoT-Like Downlink OFDMA BER vs Calculated SNR\n"
                    "(ITU-R atmospheric channel + orbital Doppler)")
    ax_ber.grid(True, which='both', linestyle='--', alpha=0.6)
    ax_ber.legend()
    plt.tight_layout()
    plt.show()

    print("\n  Simulation done")
