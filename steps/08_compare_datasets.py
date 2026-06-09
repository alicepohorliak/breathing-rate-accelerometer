import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks

# Папки з даними, результатами і графіками
data_dir = Path("data")
results_dir = Path("results")
figures_dir = Path("figures")

results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Датасети для порівняння
# Обрано три контрастні записи:
# нормальне дихання, повільне дихання та запис із кашлем/артефактами.
datasets = {
    "Нормальне дихання": "acc_normal.csv",
    "Повільне дихання": "acc_slow.csv",
    "Кашель / артефакти": "acc_cough.csv"
}

# Основні параметри обробки
fs_resampled = 50.0
lowcut = 0.1
highcut = 0.7
filter_order = 4
min_distance_seconds = 2.0

# Тривалість фрагмента для оглядового графіка, с
fragment_seconds = 30

results = []
processed_signals = []


# Визначення правильної форми слова "пік"
def peaks_word(count):
    if count == 1:
        return "пік"
    if 2 <= count <= 4:
        return "піки"
    return "піків"


# Зчитування датасету
def read_dataset(input_file):
    """
    Зчитування CSV-файлу з додатковим акселерометричним датасетом.
    Очікувані колонки: seconds, data.
    Якщо заголовків немає, файл читається як дві колонки.
    """

    df = pd.read_csv(input_file)

    if "seconds" in df.columns and "data" in df.columns:
        df = df[["seconds", "data"]].copy()
    else:
        df = pd.read_csv(input_file, header=None, names=["seconds", "data"])

    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    df["data"] = pd.to_numeric(df["data"], errors="coerce")

    df = df.dropna(subset=["seconds", "data"])
    df = df.sort_values("seconds")
    df = df.drop_duplicates(subset=["seconds"])

    # Початок часу переноситься в 0 секунд
    df["seconds"] = df["seconds"] - df["seconds"].iloc[0]

    return df.reset_index(drop=True)


# Оцінка початкової частоти дискретизації
def estimate_sampling_frequency(time_values):
    time_diff = np.diff(time_values)
    mean_dt = np.mean(time_diff)

    if mean_dt <= 0:
        return 0

    return 1 / mean_dt


# Центрування сигналу
def center_signal(signal):
    return signal - np.mean(signal)


# Ресемплінг сигналу до заданої частоти
def resample_signal(time_values, signal, target_fs):
    duration_seconds = time_values[-1] - time_values[0]

    new_time = np.arange(0, duration_seconds, 1 / target_fs)
    new_signal = np.interp(new_time, time_values, signal)

    return new_time, new_signal


# Смугова фільтрація сигналу
def bandpass_filter(signal, fs, lowcut, highcut, order):
    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered_signal = sosfiltfilt(sos, signal)

    return filtered_signal


# Адаптивний підбір prominence для різних датасетів
def calculate_prominence(filtered_signal):
    """
    Для додаткових датасетів використовується адаптивний prominence,
    тому що значення сигналу можуть мати інший масштаб, ніж основний датасет.

    Prominence визначається відносно амплітуди конкретного сигналу.
    """

    signal_std = np.std(filtered_signal)
    signal_range = np.percentile(filtered_signal, 95) - np.percentile(filtered_signal, 5)

    prominence_value = max(signal_std * 0.45, signal_range * 0.12)

    return prominence_value


# Короткий опис датасету для таблиці
def dataset_comment(dataset_name):
    comments = {
        "Нормальне дихання": "Регулярний запис із помірною частотою дихання",
        "Повільне дихання": "Запис із меншою кількістю дихальних циклів за хвилину",
        "Кашель / артефакти": "Запис із різкими коливаннями, які можуть впливати на пошук піків"
    }

    return comments.get(dataset_name, "")


# Обробка одного датасету
def process_dataset(dataset_name, file_name):
    input_file = data_dir / file_name

    if not input_file.exists():
        print(f"Файл не знайдено: {input_file}")
        return None

    # Зчитування даних
    df = read_dataset(input_file)

    time_values = df["seconds"].to_numpy()
    raw_signal = df["data"].to_numpy()

    rows_count = len(df)
    duration_seconds = time_values[-1] - time_values[0]
    duration_minutes = duration_seconds / 60
    original_fs = estimate_sampling_frequency(time_values)

    # Центрування сигналу
    centered_signal = center_signal(raw_signal)

    # Ресемплінг до 50 Гц
    resampled_time, resampled_signal = resample_signal(
        time_values,
        centered_signal,
        fs_resampled
    )

    # Смугова фільтрація
    filtered_signal = bandpass_filter(
        resampled_signal,
        fs_resampled,
        lowcut,
        highcut,
        filter_order
    )

    # Пошук піків
    distance_samples = int(min_distance_seconds * fs_resampled)
    prominence_value = calculate_prominence(filtered_signal)

    peaks, _ = find_peaks(
        filtered_signal,
        distance=distance_samples,
        prominence=prominence_value
    )

    peaks_count = len(peaks)
    breathing_rate = peaks_count / duration_minutes

    # Збереження обробленого сигналу для кожного датасету
    # Це не окремий графік, а CSV для перевірки або подальшого аналізу.
    processed_df = pd.DataFrame({
        "time_s": resampled_time,
        "centered_signal": resampled_signal,
        "filtered_signal": filtered_signal,
        "is_peak": 0
    })

    processed_df.loc[peaks, "is_peak"] = 1

    processed_csv = results_dir / f"dataset_{file_name.replace('.csv', '')}_processed.csv"
    processed_df.to_csv(processed_csv, index=False)

    # Збереження даних для спільного оглядового графіка
    processed_signals.append({
        "dataset": dataset_name,
        "time": resampled_time,
        "filtered_signal": filtered_signal,
        "peaks": peaks,
        "breathing_rate": round(breathing_rate, 2)
    })

    return {
        "dataset": dataset_name,
        "file": file_name,
        "rows_count": rows_count,
        "duration_seconds": round(duration_seconds, 2),
        "original_fs_hz": round(original_fs, 2),
        "resampled_fs_hz": fs_resampled,
        "lowcut_hz": lowcut,
        "highcut_hz": highcut,
        "min_distance_seconds": min_distance_seconds,
        "distance_samples": distance_samples,
        "prominence": round(prominence_value, 4),
        "peaks_count": peaks_count,
        "breathing_rate_bpm": round(breathing_rate, 2),
        "comment": dataset_comment(dataset_name),
        "processed_csv": processed_csv.name
    }


# Порівняння всіх обраних датасетів
for dataset_name, file_name in datasets.items():
    print()
    print(f"Обробка датасету: {dataset_name} ({file_name})")

    result = process_dataset(dataset_name, file_name)

    if result is not None:
        results.append(result)

        count = result["peaks_count"]
        word = peaks_word(count)

        print(f"Кількість рядків: {result['rows_count']}")
        print(f"Тривалість запису: {result['duration_seconds']:.2f} с")
        print(f"Початкова частота дискретизації: {result['original_fs_hz']:.2f} Гц")
        print(f"Частота після ресемплінгу: {result['resampled_fs_hz']:.2f} Гц")
        print(f"Prominence: {result['prominence']}")
        print(f"Знайдено: {count} {word}")
        print(f"Частота дихання: {result['breathing_rate_bpm']:.2f} дих./хв")


# Формування загальної таблиці результатів
results_df = pd.DataFrame(results)

output_csv = results_dir / "datasets_comparison.csv"
results_df.to_csv(output_csv, index=False)

# Формування скороченої таблиці для дипломної
summary_df = results_df[[
    "dataset",
    "file",
    "duration_seconds",
    "original_fs_hz",
    "prominence",
    "peaks_count",
    "breathing_rate_bpm",
    "comment"
]]

summary_csv = results_dir / "datasets_summary.csv"
summary_df.to_csv(summary_csv, index=False)


# Побудова графіка порівняння частоти дихання
plt.figure(figsize=(12, 6))

bars = plt.bar(
    results_df["dataset"],
    results_df["breathing_rate_bpm"]
)

plt.title("Порівняння частоти дихання для різних датасетів")
plt.xlabel("Датасет")
plt.ylabel("Частота дихання, дих./хв")
plt.grid(axis="y")
plt.tight_layout()

# Підписи значень над стовпчиками
for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

output_figure = figures_dir / "datasets_comparison.png"
plt.savefig(output_figure, dpi=300)

# Відображення графіка
plt.show()


# Побудова оглядового графіка з короткими фрагментами сигналів
# Щоб графік не був перевантажений, показуємо лише перші 30 секунд кожного запису.
fig, axes = plt.subplots(
    nrows=len(processed_signals),
    ncols=1,
    figsize=(12, 8),
    sharex=True
)

if len(processed_signals) == 1:
    axes = [axes]

for ax, item in zip(axes, processed_signals):
    time_values = item["time"]
    filtered_signal = item["filtered_signal"]
    peaks = item["peaks"]
    breathing_rate = item["breathing_rate"]
    dataset_name = item["dataset"]

    # Вибір тільки фрагмента сигналу
    fragment_mask = time_values <= fragment_seconds

    fragment_time = time_values[fragment_mask]
    fragment_signal = filtered_signal[fragment_mask]

    # Піки, які потрапляють у вибраний фрагмент
    fragment_peak_indices = [
        peak for peak in peaks
        if time_values[peak] <= fragment_seconds
    ]

    ax.plot(
        fragment_time,
        fragment_signal,
        linewidth=2,
        label="Відфільтрований сигнал"
    )

    ax.scatter(
        time_values[fragment_peak_indices],
        filtered_signal[fragment_peak_indices],
        color="red",
        s=35,
        label="Знайдені піки",
        zorder=3
    )

    ax.set_title(
        f"{dataset_name}: {breathing_rate:.2f} дих./хв",
        fontsize=12
    )

    ax.set_ylabel("Амплітуда")
    ax.grid(True)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Час, с")

plt.suptitle(
    "Фрагменти відфільтрованих сигналів для різних датасетів",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

overview_figure = figures_dir / "datasets_peaks_overview.png"
plt.savefig(overview_figure, dpi=300)

# Відображення графіка
plt.show()


# Короткий вивід результатів
print()
print("Крок 8. Порівняння різних датасетів")
print(f"Кількість датасетів для порівняння: {len(datasets)}")
print(f"Частота ресемплінгу: {fs_resampled:.2f} Гц")
print(f"Діапазон смугової фільтрації: {lowcut:.1f}–{highcut:.1f} Гц")
print(f"Мінімальна відстань між піками: {min_distance_seconds:.1f} с")
print(f"Тривалість фрагмента для оглядового графіка: {fragment_seconds} с")
print("Prominence підбирається адаптивно для кожного датасету")

print()
print("Результати порівняння:")

for _, row in results_df.iterrows():
    count = int(row["peaks_count"])
    word = peaks_word(count)

    print()
    print(f"{row['dataset']}:")
    print(f"- файл: {row['file']}")
    print(f"- тривалість: {row['duration_seconds']:.2f} с")
    print(f"- початкова частота: {row['original_fs_hz']:.2f} Гц")
    print(f"- prominence: {row['prominence']:.4f}")
    print(f"- знайдено: {count} {word}")
    print(f"- частота дихання: {row['breathing_rate_bpm']:.2f} дих./хв")
    print(f"- коментар: {row['comment']}")

print()
print("Файли результатів:")
print(f"- повна таблиця порівняння: {output_csv}")
print(f"- скорочена таблиця для дипломної: {summary_csv}")
print(f"- графік порівняння частоти: {output_figure}")
print(f"- оглядовий графік із фрагментами сигналів: {overview_figure}")