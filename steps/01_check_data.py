import pandas as pd
from pathlib import Path

# Шлях до вхідного CSV-файлу з даними акселерометра
file_path = Path("data/breathing_sitting_position.csv")

# Основні колонки, необхідні для подальшої обробки сигналу
required_columns = ["time[us]", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]

# Зчитування CSV-файлу з пропуском службових рядків
df = pd.read_csv(file_path, skiprows=2)

# Обчислення тривалості запису за часовими мітками у мікросекундах
start_time_us = df["time[us]"].iloc[0]
end_time_us = df["time[us]"].iloc[-1]
duration_s = (end_time_us - start_time_us) / 1_000_000

# Перевірка наявності потрібних колонок у файлі
missing_columns = [col for col in required_columns if col not in df.columns]

# Виведення загальної інформації про файл
print()
print("Крок 1. Перевірка вхідних даних")
print(f"Файл: {file_path}")
print(f"Кількість рядків: {df.shape[0]}")
print(f"Кількість колонок: {df.shape[1]}")
print(f"Тривалість запису: {duration_s:.2f} с")

# Виведення статусу основних колонок
print()
print("Основні колонки:")
for col in required_columns:
    status = "знайдено" if col in df.columns else "не знайдено"
    print(f"- {col}: {status}")

# Перевірка пропущених значень і виведення перших рядків
if not missing_columns:
    print()
    print("Пропущені значення в основних колонках:")
    missing_values = df[required_columns].isna().sum()

    for col, value in missing_values.items():
        print(f"- {col}: {value}")

    print()
    print("Перші 5 рядків основних даних:")
    print(df[required_columns].head())

    print()
    print("Перевірку завершено успішно.")
else:
    print()
    print("Відсутні потрібні колонки:")
    for col in missing_columns:
        print(f"- {col}")