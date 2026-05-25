"""
Консольний запуск алгоритму визначення частоти дихання
за даними акселерометра.

Файл виконує повний алгоритмічний pipeline без Streamlit-інтерфейсу:
1. зчитування CSV-даних;
2. центрування сигналів;
3. ресемплінг;
4. смугова фільтрація;
5. формування комбінованих сигналів;
6. пошук піків на PCA-компоненті;
7. розрахунок частоти дихання;
8. збереження результатів і фінального графіка.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


# Вхідні та вихідні шляхи
DATA_FILE = Path("data/breathing_sitting_position.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Основні колонки CSV-файлу
REQUIRED_COLUMNS = ["time[us]", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]

# Параметри обробки сигналів
TARGET_FS = 50.0
LOWCUT = 0.1
HIGHCUT = 0.7
FILTER_ORDER = 4

# Параметри пошуку піків
FINAL_SIGNAL = "pca_component"
MIN_DISTANCE_SECONDS = 2.0
PROMINENCE_VALUE = 2.0


def prepare_output_dirs() -> None:
    """Створює папки для результатів і графіків, якщо їх ще немає."""
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_data(file_path: Path) -> pd.DataFrame:
    """Зчитує CSV-файл, перевіряє потрібні колонки та додає час у секундах."""
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


def resample_signal(time: np.ndarray, signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
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


def calculate_magnitude_signal(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> np.ndarray:
    """Обчислює модуль вектора прискорення за трьома осями."""
    return np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)


def calculate_pca_component(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> np.ndarray:
    """Обчислює першу PCA-компоненту за трьома відфільтрованими осями."""
    signals_matrix = np.column_stack([acc_x, acc_y, acc_z])
    centered_matrix = signals_matrix - signals_matrix.mean(axis=0)

    _, _, vt = np.linalg.svd(centered_matrix, full_matrices=False)
    pca_component = centered_matrix @ vt[0]

    # Узгодження напряму PCA-компоненти з віссю Y для зручності інтерпретації.
    correlation = np.corrcoef(pca_component, acc_y)[0, 1]
    if correlation < 0:
        pca_component = -pca_component

    return pca_component


def add_combined_signals(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Додає до таблиці комбіновані сигнали magnitude_signal і pca_component."""
    combined_df = filtered_df.copy()

    acc_x = combined_df["acc_x_filtered"].to_numpy()
    acc_y = combined_df["acc_y_filtered"].to_numpy()
    acc_z = combined_df["acc_z_filtered"].to_numpy()

    combined_df["magnitude_signal"] = calculate_magnitude_signal(acc_x, acc_y, acc_z)
    combined_df["pca_component"] = calculate_pca_component(acc_x, acc_y, acc_z)

    return combined_df


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


def save_final_outputs(
    time: np.ndarray,
    signal: np.ndarray,
    peaks: np.ndarray,
    properties: dict,
    distance_samples: int,
    duration_seconds: float,
    breathing_rate: float,
) -> None:
    """Зберігає фінальні CSV-файли та графік з позначеними піками."""
    final_result_df = pd.DataFrame({
        "selected_signal": [FINAL_SIGNAL],
        "min_distance_seconds": [MIN_DISTANCE_SECONDS],
        "distance_samples": [distance_samples],
        "prominence": [PROMINENCE_VALUE],
        "duration_seconds": [round(duration_seconds, 2)],
        "peaks_count": [len(peaks)],
        "breathing_rate_bpm": [round(breathing_rate, 2)],
    })

    final_result_file = RESULTS_DIR / "final_result.csv"
    final_result_df.to_csv(final_result_file, index=False)

    final_peaks_df = pd.DataFrame({
        "peak_index": peaks,
        "time_s": time[peaks],
        "amplitude": signal[peaks],
        "prominence": properties["prominences"],
    })

    final_peaks_file = RESULTS_DIR / "final_peaks.csv"
    final_peaks_df.to_csv(final_peaks_file, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, label="PCA component", linewidth=1.5)
    plt.scatter(time[peaks], signal[peaks], color="red", label="Знайдені піки", zorder=3)

    plt.title("Фінальний результат пошуку піків на PCA-компоненті")
    plt.xlabel("Час, с")
    plt.ylabel("Амплітуда PCA-компоненти")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    final_figure = FIGURES_DIR / "final_peaks.png"
    plt.savefig(final_figure, dpi=300)
    plt.show()

    print()
    print("Файли фінального результату:")
    print(f"- фінальна таблиця: {final_result_file}")
    print(f"- координати піків: {final_peaks_file}")
    print(f"- фінальний графік: {final_figure}")


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

    # 4. Комбіновані сигнали
    combined_df = add_combined_signals(filtered_df)

    combined_file = RESULTS_DIR / "combined_signals_data.csv"
    combined_df.to_csv(combined_file, index=False)

    print()
    print("Крок 4. Порівняння окремих і комбінованих сигналів")
    print("Додано комбіновані сигнали: magnitude_signal, pca_component")
    print(f"Файл збережено: {combined_file}")

    # 5. Фінальний пошук піків на PCA-компоненті
    final_time = combined_df["time_s"].to_numpy()
    final_signal = combined_df[FINAL_SIGNAL].to_numpy()

    final_peaks, final_properties, distance_samples = detect_breathing_peaks(
        final_signal,
        TARGET_FS,
        MIN_DISTANCE_SECONDS,
        PROMINENCE_VALUE,
    )

    final_breathing_rate = calculate_breathing_rate(len(final_peaks), duration_seconds)

    print()
    print("Крок 5. Фінальний пошук піків")
    print(f"Обраний сигнал: {FINAL_SIGNAL}")
    print(f"Мінімальна відстань між піками: {MIN_DISTANCE_SECONDS:.1f} с")
    print(f"Мінімальна відстань у точках: {distance_samples}")
    print(f"Prominence: {PROMINENCE_VALUE:.1f}")
    print(f"Кількість знайдених піків: {len(final_peaks)}")

    # 6. Збереження фінального результату
    save_final_outputs(
        time=final_time,
        signal=final_signal,
        peaks=final_peaks,
        properties=final_properties,
        distance_samples=distance_samples,
        duration_seconds=duration_seconds,
        breathing_rate=final_breathing_rate,
    )

    print()
    print("Підсумковий результат")
    print(f"Обраний сигнал: {FINAL_SIGNAL}")
    print(f"Тривалість запису: {duration_seconds:.2f} с")
    print(f"Кількість знайдених піків: {len(final_peaks)}")
    print(f"Частота дихання: {final_breathing_rate:.2f} дих./хв")


if __name__ == "__main__":
    main()
