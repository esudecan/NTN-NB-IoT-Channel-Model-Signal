import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load, EarthSatellite, wgs84
from datetime import timedelta
from math import erfc
import random

try:
    import itur
    ITUR_AVAILABLE = True
except Exception:
    ITUR_AVAILABLE = False

np.random.seed(42)
random.seed(42)

CHANNEL_GRAPHICS_AVAILABLE = False
SCFDMA_GRAPHICS_AVAILABLE = True

def qpsk_mod(bits):
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): 1 - 1j,
        (1, 0): -1 + 1j,
        (1, 1): -1 - 1j
    }
    return np.array([mapping[tuple(b)] for b in bits.reshape(-1, 2)], dtype=complex) / np.sqrt(2)


def qpsk_demod(symbols):
    bits = np.zeros((len(symbols), 2), dtype=np.uint8)
    bits[:, 0] = symbols.real < 0
    bits[:, 1] = symbols.imag < 0
    return bits.reshape(-1)


def get_itur_atmospheric_loss(lat, lon, freq_ghz, elevation_deg, p_percent, antenna_diameter_m):
    el_safe = max(float(elevation_deg), 1.0)
    if ITUR_AVAILABLE:
        Ag, Ac, Ar, As, At = itur.atmospheric_attenuation_slant_path(
            lat,
            lon,
            freq_ghz,
            el_safe,
            p_percent,
            antenna_diameter_m,
            return_contributions=True,
            include_rain=True,
            include_gas=True,
            include_scintillation=True,
            include_clouds=True
        )
        return float(Ag.value), float(Ac.value), float(Ar.value), float(As.value), float(At.value)
    Ag = 0.02 / np.sin(np.radians(el_safe))
    Ac = 0.01 / np.sin(np.radians(el_safe))
    Ar = 0.15 / np.sin(np.radians(el_safe))
    As = 0.03 / np.sin(np.radians(el_safe))
    At = Ag + Ac + Ar + As
    return Ag, Ac, Ar, As, At


def compute_channel_series():
    ts = load.timescale()
    t_now = ts.now()

    tle_line1 = '1 52755U 22057AH  24128.50000000  .00000000  00000-0  00000-0 0  9997'
    tle_line2 = '2 52755  97.5000 150.0000 0010000   0.0000 360.0000 15.00000000100000'

    satellite = EarthSatellite(tle_line1, tle_line2, 'CONNECTA', ts)
    station_lat = 39.9208
    station_lon = 32.8541
    ground_station = wgs84.latlon(station_lat, station_lon)

    carrier_freq_mhz = 2000.0
    carrier_freq_ghz = carrier_freq_mhz / 1000.0
    carrier_freq_hz = carrier_freq_mhz * 1e6
    c_km_s = 299792.458

    tx_power_dbm = 30.0
    tx_gain_dbi = 20.0
    rx_gain_dbi = 0.0
    rx_sensitivity_dbm = -125.0

    itu_exceedance_percent = 1.0
    earth_station_diameter_m = 0.2

    K_BOLT = 1.380649e-23
    T_SYS = 290.0
    BW_HZ = 180e3
    noise_power_dbm = 10 * np.log10(K_BOLT * T_SYS * BW_HZ / 1e-3)

    t_future = ts.from_datetime(t_now.utc_datetime() + timedelta(days=7))
    pass_times, pass_events = satellite.find_events(ground_station, t_now, t_future, altitude_degrees=5.0)
    rise_indices = [i for i, ev in enumerate(pass_events) if ev == 0]

    diff = satellite - ground_station
    best_culm_time = None
    best_max_el = -999.0

    for pi in range(len(rise_indices)):
        ri = rise_indices[pi]
        ci = ri + 1
        si = ri + 2
        if si >= len(pass_times):
            continue
        if pass_events[ci] != 1 or pass_events[si] != 2:
            continue
        alt, _, _ = diff.at(pass_times[ci]).altaz()
        if alt.degrees > best_max_el:
            best_max_el = alt.degrees
            best_culm_time = pass_times[ci]

    if best_culm_time is None:
        if len(rise_indices) > 0:
            idx_r = rise_indices[0]
            best_culm_time = pass_times[idx_r + 1] if idx_r + 1 < len(pass_times) else pass_times[idx_r]
        else:
            best_culm_time = t_now

    t_start = ts.from_datetime(best_culm_time.utc_datetime() - timedelta(minutes=5))
    total_seconds = 600

    time_minutes_list = []
    elevation_list = []
    distance_list = []
    doppler_list = []
    fspl_list = []
    total_path_loss_list = []
    k_factor_list = []
    link_budget_list = []
    snr_list = []
    sat_lat_list = []
    sat_lon_list = []
    markov_list = []
    atm_gas_list = []
    atm_cloud_list = []
    atm_rain_list = []
    atm_scint_list = []
    atm_total_list = []

    for s in range(total_seconds):
        current_t = ts.from_datetime(t_start.utc_datetime() + timedelta(seconds=s))
        alt, _, dist_obj = diff.at(current_t).altaz()
        el_deg = alt.degrees
        d_km = dist_obj.km

        time_minutes_list.append(s / 60.0)
        elevation_list.append(el_deg)
        distance_list.append(d_km)

        sp = wgs84.subpoint(satellite.at(current_t))
        sat_lat_list.append(sp.latitude.degrees)
        sat_lon_list.append(sp.longitude.degrees)

        t_next = ts.from_datetime(current_t.utc_datetime() + timedelta(seconds=1))
        d_next = diff.at(t_next).distance().km
        v_rel = d_km - d_next
        doppler_hz = (v_rel / c_km_s) * carrier_freq_hz
        doppler_list.append(doppler_hz)

        el_s = max(el_deg, 1.0)
        fspl = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(carrier_freq_mhz)
        atm_gas, atm_cloud, atm_rain, atm_scint, atm = get_itur_atmospheric_loss(
            station_lat,
            station_lon,
            carrier_freq_ghz,
            el_s,
            itu_exceedance_percent,
            earth_station_diameter_m
        )
        total_loss = fspl + atm

        fspl_list.append(fspl)
        total_path_loss_list.append(total_loss)
        atm_gas_list.append(atm_gas)
        atm_cloud_list.append(atm_cloud)
        atm_rain_list.append(atm_rain)
        atm_scint_list.append(atm_scint)
        atm_total_list.append(atm)

        k_db = min(15.0, max(0.0, 2.0 + ((el_s - 10.0) / 80.0) * 13.0))
        k_factor_list.append(k_db)

        rx_power = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - total_loss
        snr_db = rx_power - noise_power_dbm
        link_budget_list.append(rx_power)
        snr_list.append(snr_db)

        if el_deg >= 50.0:
            markov_list.append('A')
        elif el_deg >= 25.0:
            markov_list.append('B')
        else:
            markov_list.append('C')

        #print(f"t={s/60:.2f} min | El={el_deg:.2f} deg | d={d_km:.1f} km | Dop={doppler_hz:.1f} Hz | Loss={total_loss:.2f} dB | SNR={snr_db:.2f} dB | State={markov_list[-1]}")

    return {
        'time_arr': np.array(time_minutes_list),
        'el_arr': np.array(elevation_list),
        'dist_arr': np.array(distance_list),
        'dop_arr': np.array(doppler_list),
        'fspl_arr': np.array(fspl_list),
        'loss_arr': np.array(total_path_loss_list),
        'k_arr': np.array(k_factor_list),
        'lb_arr': np.array(link_budget_list),
        'snr_arr': np.array(snr_list),
        'atm_gas_arr': np.array(atm_gas_list),
        'atm_cloud_arr': np.array(atm_cloud_list),
        'atm_rain_arr': np.array(atm_rain_list),
        'atm_scint_arr': np.array(atm_scint_list),
        'atm_total_arr': np.array(atm_total_list),
        'sat_lat_arr': np.array(sat_lat_list),
        'sat_lon_arr': np.array(sat_lon_list),
        'state_arr': np.array(markov_list),
        'rx_sensitivity_dbm': rx_sensitivity_dbm,
        'noise_power_dbm': noise_power_dbm
    }


def resample_channel_to_symbols(channel_results, num_symbols):
    valid_len = len(channel_results['time_arr'])
    idx = np.linspace(0, valid_len - 1, num_symbols).astype(int)
    out = {}
    for k, v in channel_results.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == valid_len:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def make_channel_plots(channel_results):
    time_arr = channel_results['time_arr']
    el_arr = channel_results['el_arr']
    dist_arr = channel_results['dist_arr']
    dop_arr = channel_results['dop_arr']
    fspl_arr = channel_results['fspl_arr']
    loss_arr = channel_results['loss_arr']
    k_arr = channel_results['k_arr']
    lb_arr = channel_results['lb_arr']
    snr_arr = channel_results['snr_arr']
    atm_gas_arr = channel_results['atm_gas_arr']
    atm_cloud_arr = channel_results['atm_cloud_arr']
    atm_rain_arr = channel_results['atm_rain_arr']
    atm_scint_arr = channel_results['atm_scint_arr']
    atm_total_arr = channel_results['atm_total_arr']
    state_arr = channel_results['state_arr']
    rx_sens = channel_results['rx_sensitivity_dbm']

    state_counts = {s: list(state_arr).count(s) for s in ['A', 'B', 'C']}
    STATE_BG = {'A': '#c8e6c9', 'B': '#fff9c4', 'C': '#ffcdd2'}
    STATE_LABEL = {'A': 'State A - LOS', 'B': 'State B - Rician', 'C': 'State C - Rayleigh/Shadowing'}

    def add_markov_bands(ax):
        prev = state_arr[0]
        seg = time_arr[0]
        drawn = set()
        for i in range(1, len(time_arr)):
            if state_arr[i] != prev or i == len(time_arr) - 1:
                end = time_arr[i] if i < len(time_arr) - 1 else time_arr[-1]
                lbl = STATE_LABEL[prev] if prev not in drawn else None
                ax.axvspan(seg, end, alpha=0.18, color=STATE_BG[prev], label=lbl, zorder=0)
                drawn.add(prev)
                seg = time_arr[i]
                prev = state_arr[i]

    def add_culm(ax):
        ax.axvline(5.0, color='gray', linestyle=':', linewidth=1.3, alpha=0.9, label='Max Elevation')

    def fmt(ax, title, ylabel=''):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Time (Minutes)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    if CHANNEL_GRAPHICS_AVAILABLE:
        fig1, ax = plt.subplots(figsize=(10, 5))
        fig1.canvas.manager.set_window_title('Fig 1 - Total Path Loss')
        add_markov_bands(ax)
        ax.plot(time_arr, loss_arr, color='crimson', linewidth=2, label='Total Path Loss', zorder=3)
        add_culm(ax)
        fmt(ax, '1. Total Path Loss Over Time', 'Loss (dB)')
        ax.legend()
        fig1.tight_layout()
        plt.show()

        fig_atm, ax_atm = plt.subplots(figsize=(10, 5))
        fig_atm.canvas.manager.set_window_title('Fig - ITU-R Atmospheric Loss')
        add_markov_bands(ax_atm)
        ax_atm.plot(time_arr, atm_gas_arr, linewidth=1.5, label='Gas')
        ax_atm.plot(time_arr, atm_cloud_arr, linewidth=1.5, label='Cloud')
        ax_atm.plot(time_arr, atm_rain_arr, linewidth=1.5, label='Rain')
        ax_atm.plot(time_arr, atm_scint_arr, linewidth=1.5, label='Scintillation')
        ax_atm.plot(time_arr, atm_total_arr, linewidth=2.2, label='Total')
        add_culm(ax_atm)
        fmt(ax_atm, 'ITU-R Atmospheric Attenuation Contributions', 'Loss (dB)')
        ax_atm.legend(fontsize=9)
        fig_atm.tight_layout()
        plt.show()

        fig2, ax = plt.subplots(figsize=(10, 5))
        fig2.canvas.manager.set_window_title('Fig 2 - FSPL & Slant Range')
        add_markov_bands(ax)
        ln1, = ax.plot(time_arr, fspl_arr, color='darkorange', linewidth=2, label='FSPL', zorder=3)
        ax.set_ylabel('FSPL (dB)', color='darkorange', fontsize=10)
        ax.tick_params(axis='y', labelcolor='darkorange')
        axt = ax.twinx()
        ln2, = axt.plot(time_arr, dist_arr, color='steelblue', linewidth=2, linestyle='-.', label='Slant Range', zorder=3)
        axt.set_ylabel('Slant Range (km)', color='steelblue', fontsize=10)
        axt.tick_params(axis='y', labelcolor='steelblue')
        add_culm(ax)
        fmt(ax, '2. FSPL and Slant Range Over Time')
        ax.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()], fontsize=9, loc='upper right')
        fig2.tight_layout()
        plt.show()

        fig3, ax = plt.subplots(figsize=(10, 5))
        fig3.canvas.manager.set_window_title('Fig 3 - Elevation & Markov States')
        add_markov_bands(ax)
        ax.plot(time_arr, el_arr, color='royalblue', linewidth=2, label='Elevation', zorder=3)
        ax.axhline(50.0, color='green', linestyle='--', linewidth=1.3, alpha=0.85, label='50 deg A/B')
        ax.axhline(25.0, color='orange', linestyle='--', linewidth=1.3, alpha=0.85, label='25 deg B/C')
        add_culm(ax)
        fmt(ax, '3. Elevation Angle and Markov State Boundaries', 'Elevation (deg)')
        ax.legend(fontsize=9)
        fig3.tight_layout()
        plt.show()

        fig4, ax = plt.subplots(figsize=(10, 5))
        fig4.canvas.manager.set_window_title('Fig 4 - Doppler Shift')
        add_markov_bands(ax)
        ax.plot(time_arr, dop_arr / 1e3, color='darkorchid', linewidth=2.5, label='Doppler Shift', zorder=3)
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Doppler')
        add_culm(ax)
        fmt(ax, '4. Doppler Shift', 'Doppler Frequency (kHz)')
        ax.legend(fontsize=9)
        fig4.tight_layout()
        plt.show()

        fig5, ax = plt.subplots(figsize=(10, 5))
        fig5.canvas.manager.set_window_title('Fig 5 - Rician K-Factor')
        add_markov_bands(ax)
        ax.plot(time_arr, k_arr, color='saddlebrown', linewidth=2, label='Rician K-Factor', zorder=3)
        add_culm(ax)
        fmt(ax, '5. Rician K-Factor Over Time', 'K (dB)')
        ax.legend(fontsize=9)
        fig5.tight_layout()
        plt.show()

        fig6, ax = plt.subplots(figsize=(10, 5))
        fig6.canvas.manager.set_window_title('Fig 6 - Link Budget & SNR')
        add_markov_bands(ax)
        ax.plot(time_arr, lb_arr, color='teal', linewidth=2, label='Rx Power', zorder=3)
        ax.plot(time_arr, snr_arr, color='navy', linewidth=1.5, linestyle='--', label='SNR', zorder=3)
        ax.axhline(rx_sens, color='red', linestyle='--', linewidth=2, label=f'Rx Sensitivity ({rx_sens} dBm)')
        add_culm(ax)
        fmt(ax, '6. Link Budget and SNR', 'Power / SNR (dBm / dB)')
        ax.legend(fontsize=9, loc='lower right')
        fig6.tight_layout()
        plt.show()

        fig7, ax = plt.subplots(figsize=(9, 6))
        fig7.canvas.manager.set_window_title('Fig 7 - Markov State Parameters')
        states_lbl = ['A (LOS)', 'B (Moderate)', 'C (Shadow)']
        durations = [state_counts.get(k, 0) for k in ['A', 'B', 'C']]
        pct = [100 * d / len(state_arr) for d in durations]

        def state_mean(arr, sk):
            idx = np.where(state_arr == sk)[0]
            return np.mean(arr[idx]) if len(idx) else float('nan')

        ml = [state_mean(loss_arr, k) for k in ['A', 'B', 'C']]
        ms = [state_mean(snr_arr, k) for k in ['A', 'B', 'C']]
        mk = [state_mean(k_arr, k) for k in ['A', 'B', 'C']]
        bars = ax.bar(states_lbl, durations, color=['#2e7d32', '#f9a825', '#c62828'], alpha=0.88, edgecolor='black', linewidth=0.9)
        for bar, d, p, a, b, c in zip(bars, durations, pct, ml, ms, mk):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f'{d}s ({p:.0f}%)\nAvg Loss:{a:.1f}dB\nAvg SNR:{b:.1f}dB\nAvg K:{c:.1f}dB', ha='center', va='bottom', fontsize=8.5)
        ax.set_ylim(0, max(durations) * 1.6 if max(durations) > 0 else 60)
        ax.set_ylabel('Duration (seconds)', fontsize=10)
        ax.set_xlabel('Channel State', fontsize=10)
        ax.set_title('7. Markov State Parameters - 10-min Window', fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        fig7.tight_layout()
        plt.show()


def apply_time_varying_leo_channel(tx_blocks, channel_symbol, fs, delay_samples, coherence_blocks=7):
    rx_blocks = []
    h_series = []
    phase_acc = 0.0
    prev_state = channel_symbol['state_arr'][0]
    h_shadow = 1.0 + 0j
    h_fading = 1.0 + 0j

    for s, tx_block in enumerate(tx_blocks):
        doppler_hz = channel_symbol['dop_arr'][s]
        k_factor_db = channel_symbol['k_arr'][s]
        state = channel_symbol['state_arr'][s]
        loss_db = channel_symbol['loss_arr'][s]
        t_block = np.arange(len(tx_block)) / fs

        if state == 'A':
            k_eff = max(k_factor_db, 12.0)
            shadow_extra_db = 0.0
        elif state == 'B':
            k_eff = max(k_factor_db, 3.0)
            shadow_extra_db = np.random.normal(0.0, 1.0)
        else:
            k_eff = -80.0
            shadow_extra_db = np.random.normal(6.0, 2.0)

        state_changed = (state != prev_state) or (s == 0)
        if state_changed:
            h_shadow = 10 ** (-shadow_extra_db / 20)
            prev_state = state

        # Keep fading approximately constant over a pilot/data slot so pilot-based
        # equalization remains valid for neighboring data symbols.
        regen_fading = state_changed or (coherence_blocks > 0 and (s % coherence_blocks == 0))
        if regen_fading:
            if state == 'C':
                h_fading = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2)
            else:
                k_lin = 10 ** (k_eff / 10)
                los = np.sqrt(k_lin / (k_lin + 1))
                nlos = (np.random.normal(0, 1) + 1j * np.random.normal(0, 1)) / np.sqrt(2 * (k_lin + 1))
                h_fading = los + nlos

        path_loss_amp = 10 ** (-loss_db / 20)
        phase = phase_acc + 2 * np.pi * doppler_hz * t_block
        phase_acc = phase[-1] + 2 * np.pi * doppler_hz / fs
        rx_block = tx_block * np.exp(1j * phase) * h_fading * h_shadow * path_loss_amp
        rx_blocks.append(rx_block)
        h_series.append(h_fading * h_shadow * path_loss_amp)

    rx_signal = np.concatenate(rx_blocks)
    rx_signal = np.pad(rx_signal, (delay_samples, 200), mode='constant')
    return rx_signal, np.array(h_series)


def add_awgn_measured(rx_signal, snr_db):
    sig_pwr = np.mean(np.abs(rx_signal) ** 2)
    noise_pwr = sig_pwr / (10 ** (snr_db / 10))
    noise = (np.random.normal(0, 1, len(rx_signal)) + 1j * np.random.normal(0, 1, len(rx_signal))) / np.sqrt(2)
    return rx_signal + noise * np.sqrt(noise_pwr)


def run_simulation():
    channel_results = compute_channel_series()
    make_channel_plots(channel_results)

    delta_f = 15000
    M = 12
    N = 128
    CP = 9
    fs = N * delta_f
    delay_samples = 10
    num_slots = 100
    symbols_per_slot = 7
    pilot_idx = 3
    num_data_per_slot = symbols_per_slot - 1
    total_data_symbols = num_slots * num_data_per_slot
    block_len = N + CP
    start_idx = (N // 2) - (M // 2)
    # Use a non-constant pilot in DFT-spread domain to avoid near-zero pilot FFT bins.
    pilot_bits = np.random.randint(0, 2, 2 * M, dtype=np.uint8)
    pilot_symbol = qpsk_mod(pilot_bits)
    pilot_symbol_dft = np.fft.fft(pilot_symbol, n=M)

    bits_tx = np.random.randint(0, 2, total_data_symbols * 2 * M, dtype=np.uint8)
    qpsk_data_syms = qpsk_mod(bits_tx).reshape(total_data_symbols, M)

    tx_blocks = []
    data_counter = 0
    for slot in range(num_slots):
        for sym in range(symbols_per_slot):
            if sym == pilot_idx:
                d = pilot_symbol
            else:
                d = qpsk_data_syms[data_counter]
                data_counter += 1
            D = np.fft.fft(d, n=M)
            X_shift = np.zeros(N, dtype=complex)
            X_shift[start_idx:start_idx + M] = D
            x = np.fft.ifft(np.fft.ifftshift(X_shift), n=N)
            tx_blocks.append(np.concatenate([x[-CP:], x]))

    tx_signal_base = np.concatenate(tx_blocks)
    papr_db = 10 * np.log10(np.max(np.abs(tx_signal_base) ** 2) / np.mean(np.abs(tx_signal_base) ** 2))
    channel_symbol = resample_channel_to_symbols(channel_results, len(tx_blocks))

    rx_clean, h_series = apply_time_varying_leo_channel(
        tx_blocks,
        channel_symbol,
        fs,
        delay_samples,
        coherence_blocks=symbols_per_slot
    )
    snr_for_noise = float(np.mean(channel_results['snr_arr'][channel_results['el_arr'] > 0]))
    rx_signal = add_awgn_measured(rx_clean, snr_for_noise)

    window_rx_before = np.hanning(len(rx_signal))
    spectrum_rx_before = np.fft.fftshift(np.fft.fft(rx_signal * window_rx_before, 8192))

    rx_blocks_comp = []
    idx = delay_samples
    phase_comp_acc = 0.0
    for s in range(len(tx_blocks)):
        rx_block = rx_signal[idx:idx + block_len]
        doppler_comp_hz = channel_symbol['dop_arr'][s]
        t_rx = np.arange(len(rx_block)) / fs
        phase_comp = phase_comp_acc + 2 * np.pi * doppler_comp_hz * t_rx
        rx_block_comp = rx_block * np.exp(-1j * phase_comp)
        phase_comp_acc = phase_comp[-1] + 2 * np.pi * doppler_comp_hz / fs
        rx_blocks_comp.append(rx_block_comp)
        idx += block_len

    rx_signal_comp = np.concatenate(rx_blocks_comp)
    window_rx_after = np.hanning(len(rx_signal_comp))
    spectrum_rx_after = np.fft.fftshift(np.fft.fft(rx_signal_comp * window_rx_after, 8192))

    rx_data_syms = np.zeros((total_data_symbols, M), dtype=complex)
    pre_eq_list = []
    idx_block = 0
    data_counter = 0

    for slot in range(num_slots):
        slot_blocks = []
        for sym in range(symbols_per_slot):
            rx_block_comp = rx_blocks_comp[idx_block]
            block = rx_block_comp[CP:CP + N]
            X_hat = np.fft.fftshift(np.fft.fft(block, n=N))
            #D_hat = X_hat[start_idx:start_idx + M]
            #d_hat = np.fft.ifft(D_hat, n=M)
            #slot_blocks.append(d_hat)
            D_hat = X_hat[start_idx : start_idx + M]
            slot_blocks.append(D_hat)
            idx_block += 1

        H_est = slot_blocks[pilot_idx] / (pilot_symbol_dft + 1e-12)

        for sym in range(symbols_per_slot):
            if sym != pilot_idx:
                pre_eq_freq = slot_blocks[sym]
                eq_freq = pre_eq_freq / (H_est + 1e-12)
                pre_eq_list.append(np.fft.ifft(pre_eq_freq, n=M))
                rx_data_syms[data_counter] = np.fft.ifft(eq_freq, n=M)
                data_counter += 1


    rx_syms_flat = rx_data_syms.reshape(-1)
    pre_eq_flat = np.array(pre_eq_list).reshape(-1)
    bits_rx = qpsk_demod(rx_syms_flat)[:len(bits_tx)]
    ber_one_pass = np.mean(bits_rx != bits_tx)

    snr_sorted = np.array(channel_results['snr_arr'])
    snr_eval = np.linspace(np.percentile(snr_sorted, 5), np.percentile(snr_sorted, 95), 12)
    ber_vs_snr = []
    # Theoretical references for QPSK BER (assuming SNR ~= Eb/N0).
    snr_lin = 10 ** (snr_eval / 10.0)
    ber_theory_awgn = np.array([0.5 * erfc(np.sqrt(g)) for g in snr_lin])
    ber_theory_rayleigh = 0.5 * (1.0 - np.sqrt(snr_lin / (1.0 + snr_lin)))

    for snr_db in snr_eval:
        rx_noisy = add_awgn_measured(rx_clean, snr_db)
        rx_blocks_comp_loop = []
        idx = delay_samples
        phase_comp_acc_loop = 0.0
        for s in range(len(tx_blocks)):
            rx_block = rx_noisy[idx:idx + block_len]
            doppler_comp_hz = channel_symbol['dop_arr'][s]
            t_rx = np.arange(len(rx_block)) / fs
            phase_comp_loop = phase_comp_acc_loop + 2 * np.pi * doppler_comp_hz * t_rx
            rx_block_comp = rx_block * np.exp(-1j * phase_comp_loop)
            phase_comp_acc_loop = phase_comp_loop[-1] + 2 * np.pi * doppler_comp_hz / fs
            rx_blocks_comp_loop.append(rx_block_comp)
            idx += block_len

        rx_data_syms_loop = np.zeros((total_data_symbols, M), dtype=complex)
        idx_block = 0
        data_counter = 0
        for slot in range(num_slots):
            slot_blocks = []
            for sym in range(symbols_per_slot):
                rx_block_comp = rx_blocks_comp_loop[idx_block]
                block = rx_block_comp[CP:CP + N]
                X_hat = np.fft.fftshift(np.fft.fft(block, n=N))
                D_hat = X_hat[start_idx:start_idx + M]
                slot_blocks.append(D_hat)
                idx_block += 1

            H_est = slot_blocks[pilot_idx] / (pilot_symbol_dft + 1e-12)

            for sym in range(symbols_per_slot):
                if sym != pilot_idx:
                    eq_freq_loop = slot_blocks[sym] / (H_est + 1e-12)
                    rx_data_syms_loop[data_counter] = np.fft.ifft(eq_freq_loop, n=M)
                    data_counter += 1

        bits_rx_loop = qpsk_demod(rx_data_syms_loop.reshape(-1))[:len(bits_tx)]
        ber_vs_snr.append(np.mean(bits_rx_loop != bits_tx))
        print(f"SNR={snr_db:.2f} dB | BER={ber_vs_snr[-1]:.8f}")

    time_arr = channel_results['time_arr']

    if SCFDMA_GRAPHICS_AVAILABLE:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.canvas.manager.set_window_title('SC-FDMA NTN LEO Analysis')
        fig.suptitle('SC-FDMA Uplink with ITU-R LEO Channel', fontsize=16, fontweight='bold')

        ax1 = axs[0, 0]
        ax1.plot(time_arr, channel_results['el_arr'], 'b-', linewidth=2, label='Elevation')
        ax1.set_xlabel('Time (Minutes)')
        ax1.set_ylabel('Elevation (deg)', color='b')
        ax1.grid(True, alpha=0.3)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_arr, channel_results['snr_arr'], 'g-', linewidth=2, label='SNR')
        ax1_twin.set_ylabel('SNR (dB)', color='g')
        ax1.set_title('Elevation and SNR Variation')

        ax2 = axs[0, 1]
        ax2.plot(time_arr, channel_results['dop_arr'], 'purple', linewidth=2)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_title('Doppler Curve')
        ax2.set_xlabel('Time (Minutes)')
        ax2.set_ylabel('Doppler Shift (Hz)')
        ax2.grid(True, alpha=0.5)

        ax3 = axs[0, 2]
        ax3.plot(time_arr, np.ones_like(time_arr) * papr_db, color='orange', linewidth=2)
        ax3.set_title('SC-FDMA PAPR')
        ax3.set_xlabel('Time (Minutes)')
        ax3.set_ylabel('PAPR (dB)')
        ax3.grid(True, alpha=0.5)

        ax4 = axs[1, 0]
        ax4.semilogy(snr_eval, np.array(ber_vs_snr) + 1e-7, 'ro-', linewidth=2, label='Simulated')
        ax4.semilogy(snr_eval, ber_theory_awgn + 1e-12, 'k--', linewidth=1.8, label='QPSK Theory (AWGN)')
        ax4.semilogy(snr_eval, ber_theory_rayleigh + 1e-12, 'b-.', linewidth=1.8, label='QPSK Theory (Rayleigh)')
        ax4.set_title('SC-FDMA BER vs SNR')
        ax4.set_xlabel('SNR (dB)')
        ax4.set_ylabel('BER')
        ax4.grid(True, which='both', linestyle='--', alpha=0.5)
        ax4.legend(fontsize='small')

        ax5 = axs[1, 1]
        ax5.plot(time_arr, channel_results['k_arr'], 'm-', linewidth=2)
        ax5.set_title('Rician K-Factor')
        ax5.set_xlabel('Time (Minutes)')
        ax5.set_ylabel('K (dB)')
        ax5.grid(True, alpha=0.5)

        ax6 = axs[1, 2]
        max_pts = min(5000, len(rx_syms_flat))
        ax6.scatter(pre_eq_flat[:max_pts].real, pre_eq_flat[:max_pts].imag, color='orange', alpha=0.3, s=5, label='Before EQ')
        ax6.scatter(rx_syms_flat[:max_pts].real, rx_syms_flat[:max_pts].imag, color='green', alpha=0.3, s=5, label='After EQ')
        ax6.set_title('SC-FDMA RX QPSK Constellation')
        ax6.set_xlabel('In-Phase')
        ax6.set_ylabel('Quadrature')
        ax6.set_xlim(-2, 2)
        ax6.set_ylim(-2, 2)
        ax6.grid(True, linestyle='--')
        ax6.axhline(0, color='k', lw=0.5)
        ax6.axvline(0, color='k', lw=0.5)
        ax6.legend(loc='upper right', fontsize='small')

        plt.tight_layout(pad=2.0)
        plt.show()

        fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
        fig_sp.canvas.manager.set_window_title('SC-FDMA RX Spectrum')
        f_axis_before = np.linspace(-fs / 2, fs / 2, len(spectrum_rx_before)) / 1e3
        f_axis_after = np.linspace(-fs / 2, fs / 2, len(spectrum_rx_after)) / 1e3
        ax_sp.plot(f_axis_before, 20 * np.log10(np.abs(spectrum_rx_before) / np.max(np.abs(spectrum_rx_before)) + 1e-12), label='Before Compensation')
        ax_sp.plot(f_axis_after, 20 * np.log10(np.abs(spectrum_rx_after) / np.max(np.abs(spectrum_rx_after)) + 1e-12), label='After Compensation')
        ax_sp.set_xlabel('Frequency (kHz)')
        ax_sp.set_ylabel('Magnitude (dB)')
        ax_sp.set_title('SC-FDMA RX Spectrum Before / After Doppler Compensation')
        ax_sp.grid(True)
        ax_sp.legend()
        plt.tight_layout()
        plt.show()

        fig_const1, ax_const1 = plt.subplots(figsize=(6, 6))
        fig_const1.canvas.manager.set_window_title('SC-FDMA Constellation Before EQ')
        ax_const1.scatter(pre_eq_flat[:max_pts].real, pre_eq_flat[:max_pts].imag, alpha=0.35, s=6)
        ax_const1.set_xlabel('In-Phase')
        ax_const1.set_ylabel('Quadrature')
        ax_const1.set_title('SC-FDMA RX Constellation Before Equalization')
        ax_const1.grid(True)
        ax_const1.axis('equal')
        plt.tight_layout()
        plt.show()

        fig_const2, ax_const2 = plt.subplots(figsize=(6, 6))
        fig_const2.canvas.manager.set_window_title('SC-FDMA Constellation After EQ')
        ax_const2.scatter(rx_syms_flat[:max_pts].real, rx_syms_flat[:max_pts].imag, alpha=0.35, s=6)
        ax_const2.set_xlabel('In-Phase')
        ax_const2.set_ylabel('Quadrature')
        ax_const2.set_title('SC-FDMA RX Constellation After Equalization')
        ax_const2.grid(True)
        ax_const2.axis('equal')
        plt.tight_layout()
        plt.show()

    print(f"One-pass average channel SNR = {snr_for_noise:.2f} dB")
    print(f"One-pass BER = {ber_one_pass:.8f}")
    print(f"SC-FDMA PAPR = {papr_db:.2f} dB")
    print('Simulation done')


if __name__ == '__main__':
    run_simulation()
