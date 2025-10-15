# Radar_Rescue_Simulator
UWB Rescue Rover Simulator - Advanced UI

This script provides a high-fidelity proof-of-concept for a UWB radar-based
life detection system, featuring a UI layout inspired by user-provided designs.
It operates on a single-scan basis, triggered by a button press.

Key Features:
- Professional Two-Column Layout:
    - A dedicated sidebar on the left for all controls and readouts.
    - A main panel on the right for detailed signal graphs.
- Interactive Sidebar Controls with Live Map:
    - A large, color-coded "Scan Status" (Blue for life, Orange for none).
    - A "Scan Now" button to trigger a single detection cycle.
    - A slider to set the simulated "Debris Depth".
    - "Detected Vitals" panel for Breaths/Min and Heart Rate.
    - "GPS Location" panel with simulated coordinates and a live OpenStreetMap view.
- Automatic Scenario Simulation:
    - Each scan randomly determines if a victim is present and their condition,
      providing a robust test for the detection algorithm.
- Advanced Signal Processing & Visualization:
    - Uses a Butterworth bandpass filter and `scipy.find_peaks` for accurate
      vital sign detection from the simulated signal's FFT.
    - Displays Raw Signal, Filtered Signal, and Frequency Spectrum graphs.

Libraries Used:
- numpy: For numerical operations, signal generation, and FFT.
- scipy.signal: For signal filtering and peak detection.
- matplotlib: For all plotting, widgets, and layout management.
- random: For auto-generating the victim scenario for each scan.
- urllib, PIL, io, math: For fetching and displaying map tiles from OpenStreetMap.