import pandas as pd
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

# Сигнал, який використовується для базового пошуку піків
signal_column = "acc_y_filtered"

time = df["time_s"]
signal = df[signal_column]

# Параметри пошуку піків
fs = 50.0
min_distance_seconds = 2.0
distance_samples = int(min_distance_seconds * fs)
prominence_value = 2.0

# Пошук локальних максимумів сигналу
peaks, properties = find_peaks(
    signal,
    distance=distance_samples,
    prominence=prominence_value
)

peaks_count = len(peaks)

# Формування таблиці зі знайденими піками
peaks_df = pd.DataFrame({
    "peak_index": peaks,
    "time_s": time.iloc[peaks].values,
    "amplitude": signal.iloc[peaks].values,
    "prominence": properties["prominences"]
})

# Збереження координат знайдених піків
output_csv = results_dir / "basic_peaks.csv"
peaks_df.to_csv(output_csv, index=False)

# Побудова графіка сигналу з позначеними піками
plt.figure(figsize=(12, 6))

plt.plot(time, signal, label=signal_column, linewidth=1.5)
plt.scatter(
    time.iloc[peaks],
    signal.iloc[peaks],
    color="red",
    label="Знайдені піки",
    zorder=3
)

plt.title("Пошук піків на відфільтрованому сигналі")
plt.xlabel("Час, с")
plt.ylabel("Амплітуда після фільтрації, mg")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Збереження графіка
output_figure = figures_dir / "basic_peaks.png"
plt.savefig(output_figure, dpi=300)

# Відображення графіка
plt.show()

# Короткий вивід результату
print()
print("Крок 5. Базовий пошук піків")
print(f"Сигнал для аналізу: {signal_column}")
print(f"Частота дискретизації: {fs:.2f} Гц")
print(f"Мінімальна відстань між піками: {min_distance_seconds:.1f} с")
print(f"Мінімальна відстань у точках: {distance_samples}")
print(f"Prominence: {prominence_value:.1f}")
print(f"Кількість знайдених піків: {peaks_count}")
print(f"Таблицю піків збережено: {output_csv}")
print(f"Графік збережено: {output_figure}")