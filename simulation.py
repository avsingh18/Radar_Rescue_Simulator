import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from scipy.signal import butter, filtfilt, find_peaks
import time
import random
import math
import io
from urllib.request import urlopen, Request
from PIL import Image

# --- Configuration Parameters ---
SAMPLE_RATE = 100  # Hz
DURATION = 10      # seconds
N_SAMPLES = SAMPLE_RATE * DURATION

# --- Physiological Models for Different Victim States (for simulation purposes) ---
VICTIM_STATES = {
    'Alive (Stable)': {
        'resp_rate': 0.25, 'resp_amp': 0.5,  # 15 breaths/min
        'heart_rate': 1.2, 'heart_amp': 0.1,   # 72 beats/min
        'noise': 1.0
    },
    'Alive (Stressed)': {
        'resp_rate': 0.6, 'resp_amp': 0.7,  # 36 breaths/min
        'heart_rate': 1.8, 'heart_amp': 0.15,  # 108 beats/min
        'noise': 1.2
    },
    'Critical (Faint Signal)': {
        'resp_rate': 0.15, 'resp_amp': 0.1,  # 9 breaths/min
        'heart_rate': 0.8, 'heart_amp': 0.05,  # 48 beats/min
        'noise': 0.8
    },
    'No Life Detected': {
        'resp_rate': 0, 'resp_amp': 0,
        'heart_rate': 0, 'heart_amp': 0,
        'noise': 0.7
    }
}

# --- Signal Processing Parameters ---
FILTER_LOW_CUT = 0.1   # Hz
FILTER_HIGH_CUT = 2.5  # Hz
PEAK_PROMINENCE = 0.05 # For find_peaks, relative to max FFT magnitude

# --- Map Tile Functions ---
def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to OSM tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """Convert OSM tile numbers to lat/lon of the top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def get_map_image(lat, lon, zoom):
    """Fetch a single map tile from OpenStreetMap."""
    xtile, ytile = deg2num(lat, lon, zoom)
    url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        req = Request(url, headers=headers)
        with urlopen(req) as response:
            data = response.read()
        return Image.open(io.BytesIO(data))
    except Exception as e:
        print(f"Could not download map tile: {e}")
        return None

# --- Signal Processing Functions ---
def create_bandpass_filter(lowcut, highcut, fs, order=4):
    """Creates a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = create_bandpass_filter(FILTER_LOW_CUT, FILTER_HIGH_CUT, SAMPLE_RATE)

def apply_filter(data):
    """Applies the pre-computed filter to data."""
    return filtfilt(b, a, data)

def simulate_uwb_data(state_key, depth):
    """Generates simulated UWB data based on victim state and debris depth."""
    state = VICTIM_STATES[state_key]
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    noise = np.random.randn(N_SAMPLES) * state['noise']
    attenuation_factor = np.exp(-0.35 * depth)
    signal = np.zeros(N_SAMPLES)
    if state['resp_amp'] > 0:
        respiration_signal = state['resp_amp'] * np.sin(2 * np.pi * state['resp_rate'] * t)
        heartbeat_signal = state['heart_amp'] * np.sin(2 * np.pi * state['heart_rate'] * t)
        signal = (respiration_signal + heartbeat_signal) * attenuation_factor
    return t, signal + noise

def analyze_spectrum(filtered_data, fs):
    """Analyzes the frequency spectrum to find respiration and heart rate."""
    N = len(filtered_data)
    fft_vals = np.abs(np.fft.rfft(filtered_data))
    fft_freqs = np.fft.rfftfreq(N, 1 / fs)
    min_peak_height = PEAK_PROMINENCE * np.max(fft_vals) if np.max(fft_vals) > 0 else 0
    peaks, _ = find_peaks(fft_vals, height=min_peak_height, distance=10)
    detected_resp_freq, max_resp_magnitude = 0, 0
    detected_heart_freq, max_heart_magnitude = 0, 0
    for peak_idx in peaks:
        freq, magnitude = fft_freqs[peak_idx], fft_vals[peak_idx]
        if 0.1 <= freq <= 0.7 and magnitude > max_resp_magnitude:
            max_resp_magnitude, detected_resp_freq = magnitude, freq
        if 0.7 < freq <= 2.0 and magnitude > max_heart_magnitude:
            max_heart_magnitude, detected_heart_freq = magnitude, freq
    return fft_freqs, fft_vals, detected_resp_freq * 60, detected_heart_freq * 60

# --- UI Setup ---
fig = plt.figure(figsize=(16, 8), facecolor='#f0f0f0')
fig.suptitle("Advanced UWB Rescue Rover Simulator", fontsize=20, weight='bold')

gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5], wspace=0.1)

# --- Left Sidebar ---
ax_sidebar = fig.add_subplot(gs[0, 0])
ax_sidebar.set_facecolor('#f0f0f0')
ax_sidebar.spines['top'].set_visible(False)
ax_sidebar.spines['right'].set_visible(False)
ax_sidebar.spines['bottom'].set_visible(False)
ax_sidebar.spines['left'].set_visible(False)
ax_sidebar.set_xticks([]); ax_sidebar.set_yticks([])

fig.text(0.05, 0.92, "Rescue Rover Controls", fontsize=18, weight='bold')
fig.text(0.05, 0.8, "Scan Status", fontsize=14, color='gray')
status_text = fig.text(0.05, 0.74, "STANDBY", fontsize=22, weight='bold', color='gray')

fig.text(0.05, 0.5, "Detected Vitals", fontsize=14, color='gray')
vitals_resp_text = fig.text(0.05, 0.45, "Breathing: -- Breaths/Min", fontsize=12)
vitals_heart_text = fig.text(0.05, 0.41, "Heart Rate: -- BPM", fontsize=12)

fig.text(0.05, 0.32, "GPS Location", fontsize=14, color='gray')
gps_text = fig.text(0.05, 0.28, "Coordinates: Not Acquired", fontsize=12)

ax_scan_button = fig.add_axes([0.05, 0.62, 0.2, 0.075])
scan_button = Button(ax_scan_button, 'Scan Now', color='dodgerblue', hovercolor='deepskyblue')
scan_button.label.set_color('white'); scan_button.label.set_weight('bold')

ax_slider = fig.add_axes([0.05, 0.55, 0.2, 0.03])
depth_slider = Slider(ax_slider, 'Simulated Debris Depth (m)', 0.0, 10.0, valinit=3.0, valstep=0.5)

# --- Map Setup ---
ax_map = fig.add_axes([0.05, 0.05, 0.2, 0.2])
ax_map.set_xticks([]); ax_map.set_yticks([])
JAIPUR_LAT, JAIPUR_LON = 26.9124, 75.7873
ZOOM_LEVEL = 15
map_image = get_map_image(JAIPUR_LAT, JAIPUR_LON, ZOOM_LEVEL)
if map_image:
    ax_map.imshow(map_image)
else:
    # Fallback if map download fails
    map_img = np.ones((100, 100, 3)) * 0.9
    ax_map.imshow(map_img)
victim_location_marker, = ax_map.plot([], [], 'ro', ms=10, alpha=0.8)

# --- Right Graph Panel ---
gs_graphs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 1], hspace=0.5)

ax_raw = fig.add_subplot(gs_graphs[0, 0])
ax_raw.set_title("Raw Simulated UWB Signal")
ax_raw.set_xlabel("Time (s)"); ax_raw.set_ylabel("Amplitude")
ax_raw.grid(True, linestyle='--', alpha=0.6)
raw_line, = ax_raw.plot([], [], lw=1, color='gray')
ax_raw.set_ylim(-5, 5)

ax_filtered = fig.add_subplot(gs_graphs[1, 0])
ax_filtered.set_title("Filtered Signal (Biometric Frequencies)")
ax_filtered.set_xlabel("Time (s)"); ax_filtered.set_ylabel("Amplitude")
ax_filtered.grid(True, linestyle='--', alpha=0.6)
filtered_line, = ax_filtered.plot([], [], lw=2, color='c')
ax_filtered.set_ylim(-1.0, 1.0)

ax_fft = fig.add_subplot(gs_graphs[2, 0])
ax_fft.set_title("Frequency Spectrum (FFT)")
ax_fft.set_xlabel("Frequency (Hz)"); ax_fft.set_ylabel("Magnitude")
ax_fft.grid(True, linestyle='--', alpha=0.6)
fft_line, = ax_fft.plot([], [], lw=2, color='m')
resp_peak_marker, = ax_fft.plot([], [], 'bo', ms=8, label='Respiration')
heart_peak_marker, = ax_fft.plot([], [], 'ro', ms=8, label='Heartbeat')
ax_fft.legend(loc='upper right')
ax_fft.set_xlim(0, FILTER_HIGH_CUT); ax_fft.set_ylim(0, 150)

# --- Scan Function ---
def perform_scan(event):
    """Main function to run a single scan."""
    status_text.set_text("SCANNING..."); status_text.set_color('gray')
    fig.canvas.draw_idle()
    
    victim_scenario = random.choice(list(VICTIM_STATES.keys()) + ['No Life Detected'])
    current_debris_depth = depth_slider.val
    
    t, raw_data = simulate_uwb_data(victim_scenario, current_debris_depth)
    filtered_data = apply_filter(raw_data)
    fft_freqs, fft_vals, resp_bpm, heart_bpm = analyze_spectrum(filtered_data, SAMPLE_RATE)

    if resp_bpm > 0:
        final_status, final_color = "Life Detected", 'dodgerblue'
        lat = JAIPUR_LAT + random.uniform(-0.002, 0.002)
        lon = JAIPUR_LON + random.uniform(-0.002, 0.002)
        gps_text.set_text(f"Coordinates: {lat:.5f}, {lon:.5f}")

        # Convert lat/lon to pixel coordinates on the tile
        tile_lat_top, tile_lon_left = num2deg(*deg2num(JAIPUR_LAT, JAIPUR_LON, ZOOM_LEVEL), ZOOM_LEVEL)
        tile_lat_bottom, tile_lon_right = num2deg(deg2num(JAIPUR_LAT, JAIPUR_LON, ZOOM_LEVEL)[0] + 1, deg2num(JAIPUR_LAT, JAIPUR_LON, ZOOM_LEVEL)[1] + 1, ZOOM_LEVEL)
        
        # Calculate relative position of victim on the tile
        x_pos = ((lon - tile_lon_left) / (tile_lon_right - tile_lon_left)) * 256
        y_pos = ((lat - tile_lat_top) / (tile_lat_bottom - tile_lat_top)) * 256
        victim_location_marker.set_data([x_pos], [y_pos])

    else:
        final_status, final_color = "No Life Detected", 'orange'
        gps_text.set_text("Coordinates: Not Acquired")
        victim_location_marker.set_data([], [])

    raw_line.set_data(t, raw_data); ax_raw.set_xlim(0, DURATION)
    filtered_line.set_data(t, filtered_data); ax_filtered.set_xlim(0, DURATION)
    fft_line.set_data(fft_freqs, fft_vals)

    if resp_bpm > 0:
        y_val = fft_vals[np.argmin(np.abs(fft_freqs - resp_bpm/60))]
        resp_peak_marker.set_data([resp_bpm/60], [y_val])
    else:
        resp_peak_marker.set_data([], [])
    
    if heart_bpm > 0:
        y_val = fft_vals[np.argmin(np.abs(fft_freqs - heart_bpm/60))]
        heart_peak_marker.set_data([heart_bpm/60], [y_val])
    else:
        heart_peak_marker.set_data([], [])
        
    status_text.set_text(final_status); status_text.set_color(final_color)
    vitals_resp_text.set_text(f"Breathing: {resp_bpm:.1f} Breaths/Min")
    vitals_heart_text.set_text(f"Heart Rate: {heart_bpm:.1f} BPM")
    
    log_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{log_time}] Scan Complete. Ground Truth: '{victim_scenario}', Result: {final_status}, Depth: {current_debris_depth:.1f}m, Resp: {resp_bpm:.1f}, Heart: {heart_bpm:.1f}")

    fig.canvas.draw_idle()

# --- Connect button and show plot ---
scan_button.on_clicked(perform_scan)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

