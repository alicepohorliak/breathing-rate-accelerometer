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

# Основні параметри аналізу
fs = 50.0
duration_seconds = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
duration_minutes = duration_seconds / 60

# Сигнали та параметри, які порівнюються
signals = ["acc_x_filtered", "acc_y_filtered", "acc_z_filtered"]
min_distance_values = [1.5, 2.0, 2.5, 3.0]
prominence_values = [1.0, 2.0, 3.0, 4.0]

results = []


# Визначення правильної форми слова "пік"
def peaks_word(count):
    if count == 1:
        return "пік"
    if 2 <= count <= 4:
        return "піки"
    return "піків"


# Пошук піків для різних осей та параметрів
for signal_column in signals:
    signal = df[signal_column]

    for min_distance_seconds in min_distance_values:
        distance_samples = int(min_distance_seconds * fs)

        for prominence_value in prominence_values:
            peaks, _ = find_peaks(
                signal,
                distance=distance_samples,
                prominence=prominence_value
            )

            peaks_count = len(peaks)
            breathing_rate = peaks_count / duration_minutes

            results.append({
                "signal": signal_column,
                "min_distance_seconds": min_distance_seconds,
                "distance_samples": distance_samples,
                "prominence": prominence_value,
                "peaks_count": peaks_count,
                "duration_seconds": round(duration_seconds, 2),
                "breathing_rate_bpm": round(breathing_rate, 2)
            })

# Збереження повної таблиці результатів
results_df = pd.DataFrame(results)

output_csv = results_dir / "params_comparison.csv"
results_df.to_csv(output_csv, index=False)

# Формування скороченої таблиці для базової відстані 2 секунди
selected_distance = 2.0

summary_df = results_df[
    results_df["min_distance_seconds"] == selected_distance
][[
    "signal",
    "prominence",
    "peaks_count",
    "breathing_rate_bpm"
]]

summary_csv = results_dir / "params_summary.csv"
summary_df.to_csv(summary_csv, index=False)

# Побудова графіка порівняння частоти дихання
plt.figure(figsize=(12, 6))

line_styles = {
    "acc_x_filtered": {"marker": "o", "linestyle": "-", "label": "acc_x_filtered"},
    "acc_y_filtered": {"marker": "o", "linestyle": "--", "label": "acc_y_filtered"},
    "acc_z_filtered": {"marker": "o", "linestyle": "-.", "label": "acc_z_filtered"},
}

for signal_column in signals:
    signal_data = summary_df[summary_df["signal"] == signal_column]

    plt.plot(
        signal_data["prominence"],
        signal_data["breathing_rate_bpm"],
        marker=line_styles[signal_column]["marker"],
        linestyle=line_styles[signal_column]["linestyle"],
        linewidth=2,
        markersize=8,
        label=line_styles[signal_column]["label"]
    )

plt.title("Порівняння частоти дихання для різних prominence")
plt.xlabel("Prominence")
plt.ylabel("Частота дихання, дих./хв")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Збереження графіка
output_figure = figures_dir / "params_comparison.png"
plt.savefig(output_figure, dpi=300)

# Відображення графіка
plt.show()

# Короткий вивід результату
print()
print("Крок 7. Порівняння параметрів пошуку піків")
print(f"Тривалість запису: {duration_seconds:.2f} с")
print(f"Частота дискретизації: {fs:.2f} Гц")
print(f"Проаналізовані сигнали: {', '.join(signals)}")
print(f"Значення min_distance: {min_distance_values} с")
print(f"Значення prominence: {prominence_values}")

print()
print("Результати для min_distance = 2.0 с:")

for signal_column in signals:
    signal_data = summary_df[summary_df["signal"] == signal_column]

    print()
    print(f"{signal_column}:")
    for _, row in signal_data.iterrows():
        count = int(row["peaks_count"])
        word = peaks_word(count)

        print(
            f"- prominence {row['prominence']:.1f}: "
            f"{count} {word}, "
            f"{row['breathing_rate_bpm']:.2f} дих./хв"
        )

print()
print("Файли результатів:")
print(f"- повна таблиця: {output_csv}")
print(f"- скорочена таблиця: {summary_csv}")
print(f"- графік: {output_figure}")