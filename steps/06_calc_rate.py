import pandas as pd
from pathlib import Path

# Шляхи до відфільтрованих даних і таблиці знайдених піків
filtered_data_file = Path("results/filtered_data.csv")
peaks_file = Path("results/basic_peaks.csv")

# Зчитування даних
filtered_df = pd.read_csv(filtered_data_file)
peaks_df = pd.read_csv(peaks_file)

# Визначення тривалості запису в секундах і хвилинах
start_time = filtered_df["time_s"].iloc[0]
end_time = filtered_df["time_s"].iloc[-1]

duration_seconds = end_time - start_time
duration_minutes = duration_seconds / 60

# Кількість знайдених піків приймається як кількість дихальних циклів
peaks_count = len(peaks_df)

# Розрахунок частоти дихання у диханнях за хвилину
breathing_rate = peaks_count / duration_minutes

# Формування таблиці з результатом розрахунку
result_df = pd.DataFrame({
    "signal": ["acc_y_filtered"],
    "min_distance_seconds": [2.0],
    "prominence": [2.0],
    "duration_seconds": [round(duration_seconds, 2)],
    "peaks_count": [peaks_count],
    "breathing_rate_bpm": [round(breathing_rate, 2)]
})

# Збереження результату
output_file = Path("results/basic_rate.csv")
result_df.to_csv(output_file, index=False)

# Короткий вивід результату
print()
print("Крок 6. Розрахунок частоти дихання")
print(f"Початковий час: {start_time:.2f} с")
print(f"Кінцевий час: {end_time:.2f} с")
print(f"Тривалість запису: {duration_seconds:.2f} с")
print(f"Тривалість запису: {duration_minutes:.4f} хв")
print(f"Кількість знайдених піків: {peaks_count}")
print(f"Частота дихання: {breathing_rate:.2f} дих./хв")
print(f"Результат збережено: {output_file}")