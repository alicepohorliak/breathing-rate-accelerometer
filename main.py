"""
Консольний запуск алгоритму визначення частоти дихання
за даними акселерометра.

Файл виконує повний алгоритмічний pipeline без Streamlit-інтерфейсу:
1. зчитування основного CSV-файлу;
2. центрування сигналів;
3. ресемплінг;
4. смугова фільтрація;
5. пошук піків на відфільтрованому сигналі;
6. розрахунок частоти дихання для основного запису;
7. порівняння результатів алгоритму для різних датасетів;
8. збереження результатів і графіків.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


# Вхідні та вихідні шляхи
DATA_FILE = Path("data/breathing_sitting_position.csv")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Основні колонки CSV-файлу
REQUIRED_COLUMNS = ["time[us]", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]

# Додаткові датасети для порівняння
DATASETS = {
    "Нормальне дихання": "acc_normal.csv",
    "Повільне дихання": "acc_slow.csv",
    "Кашель / артефакти": "acc_cough.csv",
}

# Параметри обробки сигналів
TARGET_FS = 50.0
LOWCUT = 0.1
HIGHCUT = 0.7
FILTER_ORDER = 4

# Параметри пошуку піків для основного запису
MAIN_SIGNAL = "acc_y_filtered"
MIN_DISTANCE_SECONDS = 2.0
PROMINENCE_VALUE = 2.0

# Тривалість фрагмента для оглядового графіка додаткових датасетів
FRAGMENT_SECONDS = 30


def prepare_output_dirs() -> None:
    """Створює папки для результатів і графіків, якщо їх ще немає."""
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def peaks_word(count: int) -> str:
    """Повертає правильну форму слова 'пік' залежно від кількості."""
    if count == 1:
        return "пік"
    if 2 <= count <= 4:
        return "піки"
    return "піків"


def load_data(file_path: Path) -> pd.DataFrame:
    """Зчитує основний CSV-файл, перевіряє потрібні колонки та додає час у секундах."""
    df = pd.read_csv(file_path, skiprows=2)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Відсутні потрібні колонки: {missing_columns}")

    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["time_s"] = (df["time[us]"] - df["time[us]"].iloc[0]) / 1_000_000

    return df


def center_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Центрує сигнали акселерометра шляхом віднімання середнього значення."""
    centered_df = df[["time_s"] + REQUIRED_COLUMNS].copy()

    centered_df["acc_x_centered"] = centered_df["acc_x[mg]"] - centered_df["acc_x[mg]"].mean()
    centered_df["acc_y_centered"] = centered_df["acc_y[mg]"] - centered_df["acc_y[mg]"].mean()
    centered_df["acc_z_centered"] = centered_df["acc_z[mg]"] - centered_df["acc_z[mg]"].mean()

    return centered_df


def resample_signal(
    time: np.ndarray,
    signal: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Переводить сигнал на рівномірну часову сітку із заданою частотою."""
    dt = 1 / fs
    time_uniform = np.arange(time[0], time[-1], dt)
    signal_resampled = np.interp(time_uniform, time, signal)

    return time_uniform, signal_resampled


def butter_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int,
) -> np.ndarray:
    """Виконує смугову фільтрацію сигналу у стабільній SOS-формі."""
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def build_filtered_dataframe(centered_df: pd.DataFrame) -> pd.DataFrame:
    """Виконує ресемплінг і фільтрацію центрованих сигналів по трьох осях."""
    time = centered_df["time_s"].to_numpy()

    time_uniform, acc_x_resampled = resample_signal(
        time,
        centered_df["acc_x_centered"].to_numpy(),
        TARGET_FS,
    )

    _, acc_y_resampled = resample_signal(
        time,
        centered_df["acc_y_centered"].to_numpy(),
        TARGET_FS,
    )

    _, acc_z_resampled = resample_signal(
        time,
        centered_df["acc_z_centered"].to_numpy(),
        TARGET_FS,
    )

    acc_x_filtered = butter_bandpass_filter(
        acc_x_resampled,
        TARGET_FS,
        LOWCUT,
        HIGHCUT,
        FILTER_ORDER,
    )

    acc_y_filtered = butter_bandpass_filter(
        acc_y_resampled,
        TARGET_FS,
        LOWCUT,
        HIGHCUT,
        FILTER_ORDER,
    )

    acc_z_filtered = butter_bandpass_filter(
        acc_z_resampled,
        TARGET_FS,
        LOWCUT,
        HIGHCUT,
        FILTER_ORDER,
    )

    return pd.DataFrame({
        "time_s": time_uniform,
        "acc_x_centered_resampled": acc_x_resampled,
        "acc_y_centered_resampled": acc_y_resampled,
        "acc_z_centered_resampled": acc_z_resampled,
        "acc_x_filtered": acc_x_filtered,
        "acc_y_filtered": acc_y_filtered,
        "acc_z_filtered": acc_z_filtered,
    })


def detect_breathing_peaks(
    signal: np.ndarray,
    fs: float,
    min_distance_seconds: float,
    prominence: float,
) -> tuple[np.ndarray, dict, int]:
    """Знаходить піки у підготовленому сигналі."""
    distance_samples = int(min_distance_seconds * fs)

    peaks, properties = find_peaks(
        signal,
        distance=distance_samples,
        prominence=prominence,
    )

    return peaks, properties, distance_samples


def calculate_breathing_rate(peaks_count: int, duration_seconds: float) -> float:
    """Розраховує частоту дихання у диханнях за хвилину."""
    if duration_seconds <= 0:
        return 0.0

    duration_minutes = duration_seconds / 60
    return peaks_count / duration_minutes


def save_main_record_outputs(
    filtered_df: pd.DataFrame,
    peaks: np.ndarray,
    properties: dict,
    distance_samples: int,
    duration_seconds: float,
    breathing_rate: float,
) -> None:
    """Зберігає результат для основного запису та графік з позначеними піками."""
    time = filtered_df["time_s"].to_numpy()
    signal = filtered_df[MAIN_SIGNAL].to_numpy()

    main_result_df = pd.DataFrame({
        "selected_signal": [MAIN_SIGNAL],
        "min_distance_seconds": [MIN_DISTANCE_SECONDS],
        "distance_samples": [distance_samples],
        "prominence": [PROMINENCE_VALUE],
        "duration_seconds": [round(duration_seconds, 2)],
        "peaks_count": [len(peaks)],
        "breathing_rate_bpm": [round(breathing_rate, 2)],
    })

    main_result_file = RESULTS_DIR / "main_record_result.csv"
    main_result_df.to_csv(main_result_file, index=False)

    main_peaks_df = pd.DataFrame({
        "peak_index": peaks,
        "time_s": time[peaks],
        "amplitude": signal[peaks],
        "prominence": properties["prominences"],
    })

    main_peaks_file = RESULTS_DIR / "main_record_peaks.csv"
    main_peaks_df.to_csv(main_peaks_file, index=False)

    plt.figure(figsize=(12, 6))

    plt.plot(
        time,
        signal,
        label=MAIN_SIGNAL,
        linewidth=1.5,
    )

    plt.scatter(
        time[peaks],
        signal[peaks],
        color="red",
        label="Знайдені піки",
        zorder=3,
    )

    plt.title("Результат пошуку піків для основного запису")
    plt.xlabel("Час, с")
    plt.ylabel("Амплітуда після фільтрації, mg")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    main_figure = FIGURES_DIR / "main_record_peaks.png"
    plt.savefig(main_figure, dpi=300)
    plt.show()

    print()
    print("Файли результату для основного запису:")
    print(f"- таблиця результату: {main_result_file}")
    print(f"- координати піків: {main_peaks_file}")
    print(f"- графік із піками: {main_figure}")


def read_external_dataset(input_file: Path) -> pd.DataFrame:
    """
    Зчитує додатковий датасет.
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

    df["seconds"] = df["seconds"] - df["seconds"].iloc[0]

    return df.reset_index(drop=True)


def estimate_sampling_frequency(time_values: np.ndarray) -> float:
    """Оцінює початкову частоту дискретизації за середнім часовим кроком."""
    time_diff = np.diff(time_values)
    mean_dt = np.mean(time_diff)

    if mean_dt <= 0:
        return 0.0

    return 1 / mean_dt


def center_signal(signal: np.ndarray) -> np.ndarray:
    """Центрує одновимірний сигнал шляхом віднімання середнього значення."""
    return signal - np.mean(signal)


def calculate_adaptive_prominence(filtered_signal: np.ndarray) -> float:
    """
    Адаптивно визначає prominence для додаткових датасетів.

    Додаткові датасети мають інший масштаб сигналу, тому фіксоване
    значення prominence = 2.0 для них не використовується.
    """
    signal_std = np.std(filtered_signal)
    signal_range = np.percentile(filtered_signal, 95) - np.percentile(filtered_signal, 5)

    prominence_value = max(signal_std * 0.45, signal_range * 0.12)

    return prominence_value


def dataset_comment(dataset_name: str) -> str:
    """Повертає короткий коментар для кожного додаткового датасету."""
    comments = {
        "Нормальне дихання": "Регулярний запис із помірною частотою дихання",
        "Повільне дихання": "Запис із меншою кількістю дихальних циклів за хвилину",
        "Кашель / артефакти": "Запис із різкими коливаннями, які можуть впливати на пошук піків",
    }

    return comments.get(dataset_name, "")


def process_external_dataset(
    dataset_name: str,
    file_name: str,
    processed_signals: list[dict],
) -> dict | None:
    """Виконує повний pipeline для одного додаткового датасету."""
    input_file = DATA_DIR / file_name

    if not input_file.exists():
        print(f"Файл не знайдено: {input_file}")
        return None

    # 1. Зчитування
    df = read_external_dataset(input_file)

    time_values = df["seconds"].to_numpy()
    raw_signal = df["data"].to_numpy()

    rows_count = len(df)
    duration_seconds = time_values[-1] - time_values[0]
    duration_minutes = duration_seconds / 60
    original_fs = estimate_sampling_frequency(time_values)

    # 2. Центрування
    centered_signal = center_signal(raw_signal)

    # 3. Ресемплінг
    resampled_time, resampled_signal = resample_signal(
        time_values,
        centered_signal,
        TARGET_FS,
    )

    # 4. Смугова фільтрація
    filtered_signal = butter_bandpass_filter(
        resampled_signal,
        TARGET_FS,
        LOWCUT,
        HIGHCUT,
        FILTER_ORDER,
    )

    # 5. Пошук піків
    distance_samples = int(MIN_DISTANCE_SECONDS * TARGET_FS)
    prominence_value = calculate_adaptive_prominence(filtered_signal)

    peaks, _ = find_peaks(
        filtered_signal,
        distance=distance_samples,
        prominence=prominence_value,
    )

    peaks_count = len(peaks)
    breathing_rate = peaks_count / duration_minutes

    # 6. Збереження обробленого сигналу
    processed_df = pd.DataFrame({
        "time_s": resampled_time,
        "centered_signal": resampled_signal,
        "filtered_signal": filtered_signal,
        "is_peak": 0,
    })

    processed_df.loc[peaks, "is_peak"] = 1

    processed_csv = RESULTS_DIR / f"dataset_{file_name.replace('.csv', '')}_processed.csv"
    processed_df.to_csv(processed_csv, index=False)

    # 7. Дані для оглядового графіка
    processed_signals.append({
        "dataset": dataset_name,
        "time": resampled_time,
        "filtered_signal": filtered_signal,
        "peaks": peaks,
        "breathing_rate": round(breathing_rate, 2),
    })

    return {
        "dataset": dataset_name,
        "file": file_name,
        "rows_count": rows_count,
        "duration_seconds": round(duration_seconds, 2),
        "original_fs_hz": round(original_fs, 2),
        "resampled_fs_hz": TARGET_FS,
        "lowcut_hz": LOWCUT,
        "highcut_hz": HIGHCUT,
        "min_distance_seconds": MIN_DISTANCE_SECONDS,
        "distance_samples": distance_samples,
        "prominence": round(prominence_value, 4),
        "peaks_count": peaks_count,
        "breathing_rate_bpm": round(breathing_rate, 2),
        "comment": dataset_comment(dataset_name),
        "processed_csv": processed_csv.name,
    }


def save_datasets_comparison_outputs(
    results_df: pd.DataFrame,
    processed_signals: list[dict],
) -> None:
    """Зберігає таблиці та графіки порівняння додаткових датасетів."""
    output_csv = RESULTS_DIR / "datasets_comparison.csv"
    results_df.to_csv(output_csv, index=False)

    summary_df = results_df[[
        "dataset",
        "file",
        "duration_seconds",
        "original_fs_hz",
        "prominence",
        "peaks_count",
        "breathing_rate_bpm",
        "comment",
    ]]

    summary_csv = RESULTS_DIR / "datasets_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Графік порівняння частоти дихання
    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        results_df["dataset"],
        results_df["breathing_rate_bpm"],
    )

    plt.title("Порівняння частоти дихання для різних датасетів")
    plt.xlabel("Датасет")
    plt.ylabel("Частота дихання, дих./хв")
    plt.grid(axis="y")
    plt.tight_layout()

    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    comparison_figure = FIGURES_DIR / "datasets_comparison.png"
    plt.savefig(comparison_figure, dpi=300)
    plt.show()

    # Оглядовий графік із короткими фрагментами сигналів
    fig, axes = plt.subplots(
        nrows=len(processed_signals),
        ncols=1,
        figsize=(12, 8),
        sharex=True,
    )

    if len(processed_signals) == 1:
        axes = [axes]

    for ax, item in zip(axes, processed_signals):
        time_values = item["time"]
        filtered_signal = item["filtered_signal"]
        peaks = item["peaks"]
        breathing_rate = item["breathing_rate"]
        dataset_name = item["dataset"]

        fragment_mask = time_values <= FRAGMENT_SECONDS

        fragment_time = time_values[fragment_mask]
        fragment_signal = filtered_signal[fragment_mask]

        fragment_peak_indices = [
            peak for peak in peaks
            if time_values[peak] <= FRAGMENT_SECONDS
        ]

        ax.plot(
            fragment_time,
            fragment_signal,
            linewidth=2,
            label="Відфільтрований сигнал",
        )

        ax.scatter(
            time_values[fragment_peak_indices],
            filtered_signal[fragment_peak_indices],
            color="red",
            s=35,
            label="Знайдені піки",
            zorder=3,
        )

        ax.set_title(
            f"{dataset_name}: {breathing_rate:.2f} дих./хв",
            fontsize=12,
        )

        ax.set_ylabel("Амплітуда")
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Час, с")

    plt.suptitle(
        "Фрагменти відфільтрованих сигналів для різних датасетів",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    overview_figure = FIGURES_DIR / "datasets_peaks_overview.png"
    plt.savefig(overview_figure, dpi=300)
    plt.show()

    print()
    print("Файли результатів порівняння датасетів:")
    print(f"- повна таблиця порівняння: {output_csv}")
    print(f"- скорочена таблиця для дипломної: {summary_csv}")
    print(f"- графік порівняння частоти: {comparison_figure}")
    print(f"- оглядовий графік із фрагментами сигналів: {overview_figure}")


def compare_external_datasets() -> None:
    """Порівнює результати алгоритму для різних додаткових датасетів."""
    results = []
    processed_signals = []

    print()
    print("Крок 6. Порівняння різних датасетів")

    for dataset_name, file_name in DATASETS.items():
        print()
        print(f"Обробка датасету: {dataset_name} ({file_name})")

        result = process_external_dataset(
            dataset_name,
            file_name,
            processed_signals,
        )

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
            print(f"Коментар: {result['comment']}")

    if not results:
        print()
        print("Не вдалося обробити жоден додатковий датасет.")
        return

    results_df = pd.DataFrame(results)

    save_datasets_comparison_outputs(
        results_df=results_df,
        processed_signals=processed_signals,
    )

    print()
    print("Підсумок порівняння датасетів:")
    print(f"Кількість датасетів: {len(DATASETS)}")
    print(f"Частота ресемплінгу: {TARGET_FS:.2f} Гц")
    print(f"Діапазон фільтрації: {LOWCUT}–{HIGHCUT} Гц")
    print(f"Мінімальна відстань між піками: {MIN_DISTANCE_SECONDS:.1f} с")
    print(f"Тривалість фрагмента для оглядового графіка: {FRAGMENT_SECONDS} с")
    print("Prominence підбирається адаптивно для кожного додаткового датасету.")


def main() -> None:
    """Запускає повний алгоритм визначення частоти дихання."""
    prepare_output_dirs()

    print()
    print("Аналіз ритму дихання за даними акселерометра")

    # 1. Зчитування вхідних даних
    df = load_data(DATA_FILE)

    print()
    print("Крок 1. Перевірка вхідних даних")
    print(f"Кількість рядків: {len(df)}")
    print(f"Тривалість сирого запису: {df['time_s'].iloc[-1]:.2f} с")

    # 2. Центрування сигналів
    centered_df = center_signals(df)

    centered_file = RESULTS_DIR / "centered_data.csv"
    centered_df.to_csv(centered_file, index=False)

    print()
    print("Крок 2. Центрування сигналів")
    print(f"Файл збережено: {centered_file}")

    # 3. Ресемплінг і фільтрація
    filtered_df = build_filtered_dataframe(centered_df)

    filtered_file = RESULTS_DIR / "filtered_data.csv"
    filtered_df.to_csv(filtered_file, index=False)

    duration_seconds = filtered_df["time_s"].iloc[-1] - filtered_df["time_s"].iloc[0]

    print()
    print("Крок 3. Ресемплінг і фільтрація сигналів")
    print(f"Частота після ресемплінгу: {TARGET_FS:.2f} Гц")
    print(f"Кількість точок після ресемплінгу: {len(filtered_df)}")
    print(f"Діапазон фільтрації: {LOWCUT}–{HIGHCUT} Гц")
    print(f"Порядок фільтра: {FILTER_ORDER}")
    print(f"Файл збережено: {filtered_file}")

    # 4. Пошук піків для основного запису
    main_time = filtered_df["time_s"].to_numpy()
    main_signal = filtered_df[MAIN_SIGNAL].to_numpy()

    main_peaks, main_properties, distance_samples = detect_breathing_peaks(
        main_signal,
        TARGET_FS,
        MIN_DISTANCE_SECONDS,
        PROMINENCE_VALUE,
    )

    main_breathing_rate = calculate_breathing_rate(
        len(main_peaks),
        duration_seconds,
    )

    print()
    print("Крок 4. Пошук піків для основного запису")
    print(f"Обраний сигнал: {MAIN_SIGNAL}")
    print(f"Мінімальна відстань між піками: {MIN_DISTANCE_SECONDS:.1f} с")
    print(f"Мінімальна відстань у точках: {distance_samples}")
    print(f"Prominence: {PROMINENCE_VALUE:.1f}")
    print(f"Кількість знайдених піків: {len(main_peaks)}")

    # 5. Збереження результату для основного запису
    save_main_record_outputs(
        filtered_df=filtered_df,
        peaks=main_peaks,
        properties=main_properties,
        distance_samples=distance_samples,
        duration_seconds=duration_seconds,
        breathing_rate=main_breathing_rate,
    )

    print()
    print("Крок 5. Підсумковий результат для основного запису")
    print(f"Обраний сигнал: {MAIN_SIGNAL}")
    print(f"Тривалість запису: {duration_seconds:.2f} с")
    print(f"Кількість знайдених піків: {len(main_peaks)}")
    print(f"Частота дихання: {main_breathing_rate:.2f} дих./хв")

    # 6. Порівняння додаткових датасетів
    compare_external_datasets()

    print()
    print("Консольний запуск алгоритму завершено.")


if __name__ == "__main__":
    main()