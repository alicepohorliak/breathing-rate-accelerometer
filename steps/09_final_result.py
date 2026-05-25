import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

# Шлях до файлу з комбінованими сигналами
input_file = Path("results/combined_signals_data.csv")

# Папки для збереження результатів і графіків
results_dir = Path("results")
figures_dir = Path("figures")

results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Зчитування даних з PCA-компонентою
df = pd.read_csv(input_file)

# Параметри фінального алгоритму
signal_column = "pca_component"

fs = 50.0
min_distance_seconds = 2.0
distance_samples = int(min_distance_seconds * fs)
prominence_value = 2.0

duration_seconds = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
duration_minutes = duration_seconds / 60

time = df["time_s"]
signal = df[signal_column]

# Пошук піків на PCA-компоненті
peaks, properties = find_peaks(
    signal,
    distance=distance_samples,
    prominence=prominence_value
)

peaks_count = len(peaks)
breathing_rate = peaks_count / duration_minutes

# Формування фінальної таблиці результату
final_result_df = pd.DataFrame({
    "selected_signal": [signal_column],
    "min_distance_seconds": [min_distance_seconds],
    "distance_samples": [distance_samples],
    "prominence": [prominence_value],
    "duration_seconds": [round(duration_seconds, 2)],
    "peaks_count": [peaks_count],
    "breathing_rate_bpm": [round(breathing_rate, 2)]
})

# Збереження фінального результату
output_result_csv = results_dir / "final_result.csv"
final_result_df.to_csv(output_result_csv, index=False)

# Формування таблиці координат знайдених піків
final_peaks_df = pd.DataFrame({
    "peak_index": peaks,
    "time_s": time.iloc[peaks].values,
    "amplitude": signal.iloc[peaks].values,
    "prominence": properties["prominences"]
})

# Збереження таблиці фінальних піків
output_peaks_csv = results_dir / "final_peaks.csv"
final_peaks_df.to_csv(output_peaks_csv, index=False)

# Побудова фінального графіка з позначеними піками
plt.figure(figsize=(12, 6))

plt.plot(
    time,
    signal,
    label="PCA component",
    linewidth=1.5
)

plt.scatter(
    time.iloc[peaks],
    signal.iloc[peaks],
    color="red",
    label="Знайдені піки",
    zorder=3
)

plt.title("Фінальний результат пошуку піків на PCA-компоненті")
plt.xlabel("Час, с")
plt.ylabel("Амплітуда PCA-компоненти")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Збереження фінального графіка
output_figure = figures_dir / "final_peaks.png"
plt.savefig(output_figure, dpi=300)

# Відображення графіка
plt.show()

# Короткий вивід результату
print()
print("Крок 9. Фінальний результат алгоритму")
print(f"Обраний сигнал: {signal_column}")
print(f"Частота дискретизації: {fs:.2f} Гц")
print(f"Мінімальна відстань між піками: {min_distance_seconds:.1f} с")
print(f"Мінімальна відстань у точках: {distance_samples}")
print(f"Prominence: {prominence_value:.1f}")
print(f"Тривалість запису: {duration_seconds:.2f} с")
print(f"Кількість знайдених піків: {peaks_count}")
print(f"Частота дихання: {breathing_rate:.2f} дих./хв")

print()
print("Файли результатів:")
print(f"- фінальна таблиця: {output_result_csv}")
print(f"- координати піків: {output_peaks_csv}")
print(f"- графік: {output_figure}")