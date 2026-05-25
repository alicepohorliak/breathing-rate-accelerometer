import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Шлях до вхідного CSV-файлу з даними акселерометра
file_path = Path("data/breathing_sitting_position.csv")

# Папка для збереження графіків
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Зчитування CSV-файлу з пропуском службових рядків
df = pd.read_csv(file_path, skiprows=2)

# Перетворення часу з мікросекунд у секунди
df["time_s"] = (df["time[us]"] - df["time[us]"].iloc[0]) / 1_000_000

# Побудова графіка сирих сигналів по трьох осях акселерометра
plt.figure(figsize=(12, 6))

plt.plot(df["time_s"], df["acc_x[mg]"], label="acc_x[mg]", linewidth=1)
plt.plot(df["time_s"], df["acc_y[mg]"], label="acc_y[mg]", linewidth=1)
plt.plot(df["time_s"], df["acc_z[mg]"], label="acc_z[mg]", linewidth=1)

plt.title("Сирі сигнали акселерометра по трьох осях")
plt.xlabel("Час, с")
plt.ylabel("Прискорення, mg")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Збереження графіка у файл
output_path = figures_dir / "raw_xyz.png"
plt.savefig(output_path, dpi=300)

# Відображення графіка
plt.show()

# Короткий вивід результату
print()
print("Крок 2. Побудова сирих сигналів")
print(f"Кількість точок: {len(df)}")
print(f"Тривалість запису: {df['time_s'].iloc[-1]:.2f} с")
print(f"Графік збережено: {output_path}")