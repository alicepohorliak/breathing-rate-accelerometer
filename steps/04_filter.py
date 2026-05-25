import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

# Шлях до файлу з центрованими даними
input_file = Path("results/centered_data.csv")

# Папки для збереження результатів і графіків
results_dir = Path("results")
figures_dir = Path("figures")

results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Зчитування центрованих даних
df = pd.read_csv(input_file)

# Параметри ресемплінгу та фільтрації
target_fs = 50.0
lowcut = 0.1
highcut = 0.7
order = 4

# Формування рівномірної часової сітки
time = df["time_s"].to_numpy()
dt = 1 / target_fs
time_uniform = np.arange(time[0], time[-1], dt)

# Інтерполяція центрованих сигналів на рівномірну часову сітку
acc_x_resampled = np.interp(time_uniform, time, df["acc_x_centered"])
acc_y_resampled = np.interp(time_uniform, time, df["acc_y_centered"])
acc_z_resampled = np.interp(time_uniform, time, df["acc_z_centered"])

# Смугова фільтрація сигналу у стабільній SOS-формі
def butter_bandpass_filter(signal, fs, lowcut, highcut, order):
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)

# Фільтрація сигналів по трьох осях
acc_x_filtered = butter_bandpass_filter(acc_x_resampled, target_fs, lowcut, highcut, order)
acc_y_filtered = butter_bandpass_filter(acc_y_resampled, target_fs, lowcut, highcut, order)
acc_z_filtered = butter_bandpass_filter(acc_z_resampled, target_fs, lowcut, highcut, order)

# Формування таблиці з відфільтрованими сигналами
filtered_df = pd.DataFrame({
    "time_s": time_uniform,
    "acc_x_centered_resampled": acc_x_resampled,
    "acc_y_centered_resampled": acc_y_resampled,
    "acc_z_centered_resampled": acc_z_resampled,
    "acc_x_filtered": acc_x_filtered,
    "acc_y_filtered": acc_y_filtered,
    "acc_z_filtered": acc_z_filtered,
})

# Збереження відфільтрованих даних
output_csv = results_dir / "filtered_data.csv"
filtered_df.to_csv(output_csv, index=False)

# Побудова графіка відфільтрованих сигналів
plt.figure(figsize=(12, 6))

plt.plot(filtered_df["time_s"], filtered_df["acc_x_filtered"], label="acc_x_filtered", linewidth=1.5)
plt.plot(filtered_df["time_s"], filtered_df["acc_y_filtered"], label="acc_y_filtered", linewidth=1.5)
plt.plot(filtered_df["time_s"], filtered_df["acc_z_filtered"], label="acc_z_filtered", linewidth=1.5)

plt.title("Відфільтровані сигнали акселерометра")
plt.xlabel("Час, с")
plt.ylabel("Амплітуда після фільтрації, mg")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Збереження графіка
output_figure = figures_dir / "filtered_xyz.png"
plt.savefig(output_figure, dpi=300)

# Відображення графіка
plt.show()

# Короткий вивід результату
print()
print("Крок 4. Фільтрація сигналів")
print(f"Початкова кількість точок: {len(time)}")
print(f"Кількість точок після ресемплінгу: {len(time_uniform)}")
print(f"Частота після ресемплінгу: {target_fs:.2f} Гц")
print(f"Діапазон фільтрації: {lowcut}–{highcut} Гц")
print(f"Порядок фільтра: {order}")
print(f"Відфільтровані дані збережено: {output_csv}")
print(f"Графік збережено: {output_figure}")