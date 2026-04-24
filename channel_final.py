
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import itur
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
from matplotlib.gridspec import GridSpec

np.random.seed(42)

ITUR_AVAILABLE = True

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

# --- Figure 1: Path Loss ---

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

# Figure: ITU-R atmospheric contributions
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

#FSPL vs Slant Range
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

# --- Figure 2: Elevation & Markov States---

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

# --- Figure 3: Doppler ---

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

# --- Figure 4: Rician K ---

fig5, ax = plt.subplots(figsize=(10, 5))
fig5.canvas.manager.set_window_title('Fig 5 – Rician K-Factor')
add_markov_bands(ax)
ax.plot(time_arr, k_arr, color='saddlebrown', linewidth=2, label='Rician K-Factor (dB)', zorder=3)
add_culm(ax)
fmt(ax, '5. Rician K-Factor Over Time', ylabel='K (dB)')
ax.legend(fontsize=9)
fig5.tight_layout()
plt.show()

# --- Figure 5: Link Budget ---

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

# --- Figure 6: Markov state bar ---

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

# --- Figure 7: Ground track ---

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

print("\n  Simulation done")
