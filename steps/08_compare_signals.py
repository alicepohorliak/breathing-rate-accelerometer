import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# Шлях до файлу з відфільтрованими даними
input_file = Path("results/filtered_data.csv")

# Папки для збереження результатів і графіків
results_dir = Path("results")
figures_dir = Path("figures")

results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Зчитування відфільтрованих даних
df = pd.read_csv(input_file)

# Основні параметри пошуку піків
fs = 50.0
duration_seconds = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
duration_minutes = duration_seconds / 60

min_distance_seconds = 2.0
distance_samples = int(min_distance_seconds * fs)
prominence_value = 2.0

# Отримання відфільтрованих сигналів по трьох осях
acc_x = df["acc_x_filtered"].to_numpy()
acc_y = df["acc_y_filtered"].to_numpy()
acc_z = df["acc_z_filtered"].to_numpy()

# Обчислення модуля вектора прискорення
magnitude_signal = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

# Обчислення першої PCA-компоненти
X = np.column_stack([acc_x, acc_y, acc_z])
X_centered = X - X.mean(axis=0)

_, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
pca_component = X_centered @ Vt[0]

# Узгодження напряму PCA-компоненти з віссю Y
correlation = np.corrcoef(pca_component, acc_y)[0, 1]

if correlation < 0:
    pca_component = -pca_component

# Додавання комбінованих сигналів до таблиці
df["magnitude_signal"] = magnitude_signal
df["pca_component"] = pca_component

# Сигнали, які порівнюються між собою
signals = {
    "acc_y_filtered": df["acc_y_filtered"],
    "acc_z_filtered": df["acc_z_filtered"],
    "magnitude_signal": df["magnitude_signal"],
    "pca_component": df["pca_component"]
}

results = []


# Визначення правильної форми слова "пік"
def peaks_word(count):
    if count == 1:
        return "пік"
    if 2 <= count <= 4:
        return "піки"
    return "піків"


# Пошук піків для кожного типу сигналу
for signal_name, signal in signals.items():
    peaks, _ = find_peaks(
        signal,
        distance=distance_samples,
        prominence=prominence_value
    )

    peaks_count = len(peaks)
    breathing_rate = peaks_count / duration_minutes

    results.append({
        "signal": signal_name,
        "min_distance_seconds": min_distance_seconds,
        "prominence": prominence_value,
        "peaks_count": peaks_count,
        "duration_seconds": round(duration_seconds, 2),
        "breathing_rate_bpm": round(breathing_rate, 2)
    })

# Збереження таблиці порівняння сигналів
results_df = pd.DataFrame(results)

output_csv = results_dir / "signals_comparison.csv"
results_df.to_csv(output_csv, index=False)

# Збереження таблиці з комбінованими сигналами
output_data_csv = results_dir / "combined_signals_data.csv"
df.to_csv(output_data_csv, index=False)

# Побудова графіка порівняння частоти дихання
plt.figure(figsize=(12, 6))

plt.bar(
    results_df["signal"],
    results_df["breathing_rate_bpm"]
)

plt.title("Порівняння частоти дихання для різних типів сигналу")
plt.xlabel("Тип сигналу")
plt.ylabel("Частота дихання, дих./хв")
plt.grid(axis="y")
plt.tight_layout()

# Збереження графіка
output_figure = figures_dir / "signals_comparison.png"
plt.savefig(output_figure, dpi=300)

# Відображення графіка
plt.show()

# Короткий вивід результату
print()
print("Крок 8. Порівняння окремих і комбінованих сигналів")
print(f"Тривалість запису: {duration_seconds:.2f} с")
print(f"Частота дискретизації: {fs:.2f} Гц")
print(f"Мінімальна відстань між піками: {min_distance_seconds:.1f} с")
print(f"Prominence: {prominence_value:.1f}")

print()
print("Результати порівняння:")
for _, row in results_df.iterrows():
    count = int(row["peaks_count"])
    word = peaks_word(count)

    print(
        f"- {row['signal']}: "
        f"{count} {word}, "
        f"{row['breathing_rate_bpm']:.2f} дих./хв"
    )

print()
print("Файли результатів:")
print(f"- таблиця порівняння: {output_csv}")
print(f"- дані з комбінованими сигналами: {output_data_csv}")
print(f"- графік: {output_figure}")