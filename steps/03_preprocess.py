import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Шлях до вхідного CSV-файлу з даними акселерометра
file_path = Path("data/breathing_sitting_position.csv")

# Папки для збереження результатів і графіків
results_dir = Path("results")
figures_dir = Path("figures")

results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Зчитування CSV-файлу з пропуском службових рядків
df = pd.read_csv(file_path, skiprows=2)

# Перетворення часу з мікросекунд у секунди
df["time_s"] = (df["time[us]"] - df["time[us]"].iloc[0]) / 1_000_000

# Основні колонки акселерометра, які використовуються для обробки
acc_columns = ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]

# Формування таблиці з часом і значеннями прискорення
processed_df = df[["time_s"] + acc_columns].copy()

# Центрування сигналів шляхом віднімання середнього значення кожної осі
processed_df["acc_x_centered"] = processed_df["acc_x[mg]"] - processed_df["acc_x[mg]"].mean()
processed_df["acc_y_centered"] = processed_df["acc_y[mg]"] - processed_df["acc_y[mg]"].mean()
processed_df["acc_z_centered"] = processed_df["acc_z[mg]"] - processed_df["acc_z[mg]"].mean()

# Збереження центрованих даних
output_csv = results_dir / "centered_data.csv"
processed_df.to_csv(output_csv, index=False)

# Побудова графіка центрованих сигналів
plt.figure(figsize=(12, 6))

plt.plot(processed_df["time_s"], processed_df["acc_x_centered"], label="acc_x_centered", linewidth=1)
plt.plot(processed_df["time_s"], processed_df["acc_y_centered"], label="acc_y_centered", linewidth=1)
plt.plot(processed_df["time_s"], processed_df["acc_z_centered"], label="acc_z_centered", linewidth=1)

plt.title("Центровані сигнали акселерометра")
plt.xlabel("Час, с")
plt.ylabel("Відхилення прискорення, mg")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Збереження графіка
output_figure = figures_dir / "centered_xyz.png"
plt.savefig(output_figure, dpi=300)

# Відображення графіка
plt.show()

# Короткий вивід результату
print()
print("Крок 3. Попередня обробка сигналів")
print("Середні значення до центрування:")
print(f"- acc_x: {processed_df['acc_x[mg]'].mean():.3f} mg")
print(f"- acc_y: {processed_df['acc_y[mg]'].mean():.3f} mg")
print(f"- acc_z: {processed_df['acc_z[mg]'].mean():.3f} mg")

print()
print("Середні значення після центрування:")
print(f"- acc_x_centered: {processed_df['acc_x_centered'].mean():.6f} mg")
print(f"- acc_y_centered: {processed_df['acc_y_centered'].mean():.6f} mg")
print(f"- acc_z_centered: {processed_df['acc_z_centered'].mean():.6f} mg")

print()
print(f"Оброблені дані збережено: {output_csv}")
print(f"Графік збережено: {output_figure}")