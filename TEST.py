
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

np.random.seed(42)

def simulate_trading_volume(days=126, mean=1000, std=200):
    return np.random.normal(mean, std, days)

def smooth_trading_volume(volume, window_length=7, polyorder=2):
    return savgol_filter(volume, window_length, polyorder)

def compute_weekly_totals(volume, days_per_week=5):
    weekly_totals = [np.sum(volume[i:i+days_per_week]) for i in range(0, len(volume), days_per_week)]
    return np.array(weekly_totals)

def detect_high_volume_periods(volume, threshold_factor=1.5,min_consecutive_days=3):
    avg_volum =np.mean(volume)
    high_volume_threshold=threshold_factor * avg_volume
    high_volume_indices=np.where(volume > high_volume_threshold)[0]
    high_volume_periods []
   
    start = 0
    while start < len(high_volume_indices):
        end = start
        while end + 1 < len(high_volume_indices) and high_volume_indices[end +1] == high_volume_indices[end] + 1:
            end += 1
        if end - start+1>= min_consecutive_days:
            high_volume_periods.append((high_volume_indices[start], high_volume_indices[end]))
        start = end + 1
   
    return high_volume_periods

trading_volume = simulate_trading_volume()
smoothed_volume = smooth_trading_volume(trading_volume)
weekly_totals = compute_weekly_totals(smoothed_volume)
high_volume_periods = detect_high_volume_periods(trading_volume)

plt.figure(figsize=(12, 6))

plt.plot(trading_volume, label="Original Trading Volume", color="blue")
plt.plot(smoothed_volume, label="Smoothed Trading Volume", color="green")

weekly_x = np.arange(0, len(trading_volume), 5)[:len(weekly_totals)]
plt.plot(weekly_x, weekly_totals, label="Weekly Totals", marker='o', color="red", linestyle='None')

for period in high_volume_periods:
    plt.axvspan(period[0], period[1], color='yellow', alpha=0.3, label="High Volume Period" if period == high_volume_periods[0] else "")

plt.legend()
plt.xlabel("Trading Days")
plt.ylabel("Volume")
plt.title("Stock Trading Volume Analysis")
plt.show()

