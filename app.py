import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, find_peaks
from html import escape


st.set_page_config(
    page_title="Аналіз ритму дихання",
    layout="wide"
)

# Візуальні налаштування інтерфейсу
st.markdown(
    """
    <style>
    :root {
        --accent-blue: #60a5fa;
        --accent-blue-soft: rgba(96, 165, 250, 0.14);
        --accent-blue-border: rgba(96, 165, 250, 0.28);
        --accent-green: #74e0a5;
        --accent-green-soft: rgba(51, 191, 122, 0.14);
        --accent-green-border: rgba(51, 191, 122, 0.28);
        --card-radius: 16px;
    }

    .main-title {
        font-size: 34px;
        line-height: 1.16;
        font-weight: 760;
        margin-bottom: 12px;
        letter-spacing: -0.03em;
        color: inherit;
    }

    .main-subtitle {
        font-size: 18px;
        line-height: 1.55;
        color: inherit;
        opacity: 0.74;
        max-width: 1900px;
        margin-bottom: 34px;
    }

    .step-label {
        display: inline-block;
        padding: 7px 13px;
        margin-bottom: 12px;
        border-radius: 999px;
        background: var(--accent-blue-soft);
        color: var(--accent-blue);
        border: 1px solid var(--accent-blue-border);
        font-size: 13px;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .step-title {
        font-size: 28px;
        line-height: 1.25;
        font-weight: 720;
        margin: 0 0 10px 0;
        letter-spacing: -0.02em;
        color: inherit;
    }

    .step-description {
        font-size: 17px;
        line-height: 1.55;
        color: inherit;
        opacity: 0.76;
        margin-bottom: 24px;
        max-width: 1900px;
    }

    .metric-card {
        padding: 18px 18px 16px 18px;
        border-radius: var(--card-radius);
        background: color-mix(in srgb, var(--secondary-background-color, #111827) 92%, var(--background-color, #0b0f16) 8%);
        border: 1px solid color-mix(in srgb, var(--text-color, #f8fafc) 12%, transparent);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10);
        min-height: 104px;
    }

    .compact-metric-card {
        padding: 16px 18px 15px 18px;
        border-radius: 14px;
        background: color-mix(in srgb, var(--secondary-background-color, #111827) 92%, var(--background-color, #0b0f16) 8%);
        border: 1px solid color-mix(in srgb, var(--text-color, #f8fafc) 12%, transparent);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        min-height: 92px;
    }

    .compact-metric-label {
        font-size: 13.5px;
        color: #aab6c5;
        margin-bottom: 8px;
        font-weight: 700;
        line-height: 1.25;
    }

    .compact-metric-value {
        font-size: 25px;
        line-height: 1.12;
        font-weight: 730;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }

    .metric-label {
        font-size: 13px;
        color: #aab6c5;
        margin-bottom: 8px;
        font-weight: 650;
    }

    .metric-value {
        font-size: 29px;
        line-height: 1.1;
        font-weight: 720;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }

    .section-title-small {
        font-size: 21px;
        font-weight: 720;
        margin: 20px 0 10px 0;
        color: inherit;
    }

    .status-card {
        padding: 14px 16px;
        border-radius: 14px;
        background: var(--accent-green-soft);
        border: 1px solid var(--accent-green-border);
        color: var(--accent-green);
        font-size: 15px;
        font-weight: 650;
        margin-top: 16px;
    }

    .table-card {
        padding: 14px 14px 8px 14px;
        border-radius: var(--card-radius);
        background: #111827;
        border: 1px solid rgba(96, 165, 250, 0.16);
    }

    table.custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    table.custom-table th {
        text-align: left;
        padding: 10px 12px;
        color: #aab6c5;
        font-weight: 700;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }

    table.custom-table td {
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        color: #f8fafc;
        font-weight: 600;
    }

    table.custom-table tr:last-child td {
        border-bottom: none;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background: var(--accent-green-soft);
        color: var(--accent-green);
        font-size: 12px;
        font-weight: 750;
        border: 1px solid var(--accent-green-border);
    }

    .zero-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background: var(--accent-blue-soft);
        color: var(--accent-blue);
        font-size: 12px;
        font-weight: 750;
        border: 1px solid var(--accent-blue-border);
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(96, 165, 250, 0.16) !important;
        border-radius: 16px !important;
        background: #111827 !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        background: #111827 !important;
        color: #f8fafc !important;
        font-weight: 650 !important;
    }

    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        background: #111827 !important;
    }

    [data-testid="stDataFrame"] {
        background: #111827 !important;
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Картка для коротких числових показників
def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Компактна картка для параметрів, де багато показників в одному рядку
def compact_metric_card(label, value):
    st.markdown(
        f"""
        <div class="compact-metric-card">
            <div class="compact-metric-label">{label}</div>
            <div class="compact-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Компактна HTML-таблиця для невеликих блоків даних
def custom_table(headers, rows):
    header_html = "".join([f"<th>{header}</th>" for header in headers])
    rows_html = ""

    for row in rows:
        cells_html = "".join([f"<td>{cell}</td>" for cell in row])
        rows_html += f"<tr>{cells_html}</tr>"

    st.markdown(
        f"""
        <div class="table-card">
            <table class="custom-table">
                <thead>
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )


def dataframe_to_custom_table(dataframe, columns=None, formatters=None, max_rows=None):
    if columns is None:
        columns = list(dataframe.columns)

    if formatters is None:
        formatters = {}

    display_df = dataframe[columns].copy()

    if max_rows is not None:
        display_df = display_df.head(max_rows)

    rows = []

    for _, row in display_df.iterrows():
        row_cells = []

        for column in columns:
            value = row[column]

            if column in formatters:
                value = formatters[column](value)
            elif isinstance(value, float):
                value = f"{value:.2f}"
            else:
                value = str(value)

            row_cells.append(escape(str(value)))

        rows.append(row_cells)

    custom_table(columns, rows)


def green_status(text):
    st.markdown(
        f'<div class="status-card">{text}</div>',
        unsafe_allow_html=True
    )


# Основні налаштування
default_file = Path("data/breathing_sitting_position.csv")
required_columns = ["time[us]", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]
FILTER_ORDER = 4


# Зчитування CSV-файлу
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, skiprows=2)
    df["time_s"] = (df["time[us]"] - df["time[us]"].iloc[0]) / 1_000_000
    return df


# Центрування сигналів акселерометра
def center_signals(df):
    acc_columns = ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]
    centered_df = df[["time_s"] + acc_columns].copy()

    centered_df["acc_x_centered"] = centered_df["acc_x[mg]"] - centered_df["acc_x[mg]"].mean()
    centered_df["acc_y_centered"] = centered_df["acc_y[mg]"] - centered_df["acc_y[mg]"].mean()
    centered_df["acc_z_centered"] = centered_df["acc_z[mg]"] - centered_df["acc_z[mg]"].mean()

    return centered_df


# Смугова фільтрація сигналу у SOS-формі
def butter_bandpass_filter(signal, fs, lowcut, highcut, order):
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


# Ресемплінг і фільтрація центрованих сигналів
def filter_signals(centered_df, target_fs, lowcut, highcut, order):
    time = centered_df["time_s"].to_numpy()
    dt = 1 / target_fs
    time_uniform = np.arange(time[0], time[-1], dt)

    acc_x_resampled = np.interp(time_uniform, time, centered_df["acc_x_centered"])
    acc_y_resampled = np.interp(time_uniform, time, centered_df["acc_y_centered"])
    acc_z_resampled = np.interp(time_uniform, time, centered_df["acc_z_centered"])

    acc_x_filtered = butter_bandpass_filter(acc_x_resampled, target_fs, lowcut, highcut, order)
    acc_y_filtered = butter_bandpass_filter(acc_y_resampled, target_fs, lowcut, highcut, order)
    acc_z_filtered = butter_bandpass_filter(acc_z_resampled, target_fs, lowcut, highcut, order)

    filtered_df = pd.DataFrame({
        "time_s": time_uniform,
        "acc_x_centered_resampled": acc_x_resampled,
        "acc_y_centered_resampled": acc_y_resampled,
        "acc_z_centered_resampled": acc_z_resampled,
        "acc_x_filtered": acc_x_filtered,
        "acc_y_filtered": acc_y_filtered,
        "acc_z_filtered": acc_z_filtered,
    })

    return filtered_df


# Пошук піків у вибраному сигналі
def detect_peaks(signal, fs, min_distance_seconds, prominence_value):
    distance_samples = int(min_distance_seconds * fs)

    peaks, properties = find_peaks(
        signal,
        distance=distance_samples,
        prominence=prominence_value
    )

    return peaks, properties, distance_samples


# Порівняння параметрів пошуку піків
def compare_peak_params(filtered_df, fs, min_distance_values, prominence_values):
    duration_seconds = filtered_df["time_s"].iloc[-1] - filtered_df["time_s"].iloc[0]
    duration_minutes = duration_seconds / 60

    signals = ["acc_x_filtered", "acc_y_filtered", "acc_z_filtered"]
    results = []

    for signal_column in signals:
        signal = filtered_df[signal_column]

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

    results_df = pd.DataFrame(results)
    return results_df


# Зчитування додаткового датасету
def read_external_dataset(input_file):
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

    df["seconds"] = df["seconds"] - df["seconds"].iloc[0]

    return df.reset_index(drop=True)


# Оцінка початкової частоти дискретизації
def estimate_sampling_frequency(time_values):
    time_diff = np.diff(time_values)
    mean_dt = np.mean(time_diff)

    if mean_dt <= 0:
        return 0

    return 1 / mean_dt


# Центрування одновимірного сигналу
def center_single_signal(signal):
    return signal - np.mean(signal)


# Ресемплінг одновимірного сигналу
def resample_single_signal(time_values, signal, target_fs):
    duration_seconds = time_values[-1] - time_values[0]

    new_time = np.arange(0, duration_seconds, 1 / target_fs)
    new_signal = np.interp(new_time, time_values, signal)

    return new_time, new_signal


# Адаптивний підбір prominence для різних датасетів
def calculate_adaptive_prominence(filtered_signal):
    """
    Для додаткових датасетів prominence підбирається адаптивно,
    тому що ці файли мають інший масштаб сигналу, ніж основний запис.
    """

    signal_std = np.std(filtered_signal)
    signal_range = np.percentile(filtered_signal, 95) - np.percentile(filtered_signal, 5)

    prominence_value = max(signal_std * 0.45, signal_range * 0.12)

    return prominence_value


# Короткий опис датасету
def dataset_comment(dataset_name):
    comments = {
        "Нормальне дихання": "Регулярний запис із помірною частотою дихання",
        "Повільне дихання": "Запис із меншою кількістю дихальних циклів за хвилину",
        "Кашель / артефакти": "Запис із різкими коливаннями, які можуть впливати на пошук піків"
    }

    return comments.get(dataset_name, "")


# Обробка одного додаткового датасету
def process_external_dataset(dataset_name, file_name, target_fs, lowcut, highcut, order, min_distance_seconds):
    input_file = Path("data") / file_name

    if not input_file.exists():
        return None

    df = read_external_dataset(input_file)

    time_values = df["seconds"].to_numpy()
    raw_signal = df["data"].to_numpy()

    rows_count = len(df)
    duration_seconds = time_values[-1] - time_values[0]
    duration_minutes = duration_seconds / 60
    original_fs = estimate_sampling_frequency(time_values)

    centered_signal = center_single_signal(raw_signal)

    resampled_time, resampled_signal = resample_single_signal(
        time_values,
        centered_signal,
        target_fs
    )

    filtered_signal = butter_bandpass_filter(
        resampled_signal,
        target_fs,
        lowcut,
        highcut,
        order
    )

    distance_samples = int(min_distance_seconds * target_fs)
    prominence_value = calculate_adaptive_prominence(filtered_signal)

    peaks, _ = find_peaks(
        filtered_signal,
        distance=distance_samples,
        prominence=prominence_value
    )

    peaks_count = len(peaks)

    if duration_minutes > 0:
        breathing_rate = peaks_count / duration_minutes
    else:
        breathing_rate = 0

    return {
        "dataset": dataset_name,
        "file": file_name,
        "rows_count": rows_count,
        "duration_seconds": round(duration_seconds, 2),
        "original_fs_hz": round(original_fs, 2),
        "resampled_fs_hz": target_fs,
        "prominence": round(prominence_value, 4),
        "peaks_count": peaks_count,
        "breathing_rate_bpm": round(breathing_rate, 2),
        "comment": dataset_comment(dataset_name),
        "time": resampled_time,
        "filtered_signal": filtered_signal,
        "peaks": peaks
    }


# Порівняння різних датасетів
def compare_external_datasets(target_fs, lowcut, highcut, order, min_distance_seconds):
    datasets = {
        "Нормальне дихання": "acc_normal.csv",
        "Повільне дихання": "acc_slow.csv",
        "Кашель / артефакти": "acc_cough.csv"
    }

    results = []
    processed_signals = []

    for dataset_name, file_name in datasets.items():
        result = process_external_dataset(
            dataset_name,
            file_name,
            target_fs,
            lowcut,
            highcut,
            order,
            min_distance_seconds
        )

        if result is not None:
            results.append({
                "dataset": result["dataset"],
                "file": result["file"],
                "rows_count": result["rows_count"],
                "duration_seconds": result["duration_seconds"],
                "original_fs_hz": result["original_fs_hz"],
                "resampled_fs_hz": result["resampled_fs_hz"],
                "prominence": result["prominence"],
                "peaks_count": result["peaks_count"],
                "breathing_rate_bpm": result["breathing_rate_bpm"],
                "comment": result["comment"]
            })

            processed_signals.append({
                "dataset": result["dataset"],
                "time": result["time"],
                "filtered_signal": result["filtered_signal"],
                "peaks": result["peaks"],
                "breathing_rate": result["breathing_rate_bpm"]
            })

    results_df = pd.DataFrame(results)

    return results_df, processed_signals


# Заголовок сторінки
st.markdown(
    """
    <div class="main-title">Аналіз ритму дихання за даними акселерометра</div>
    <div class="main-subtitle">
        Інтерактивна демонстрація поетапної обробки сигналів акселерометра
        для визначення частоти дихання методом підрахунку піків.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Налаштування даних")

# Вибір джерела даних
use_default_file = st.sidebar.checkbox(
    "Використати стандартний CSV-файл",
    value=True
)

uploaded_file = None

if not use_default_file:
    uploaded_file = st.sidebar.file_uploader(
        "Завантажити CSV-файл",
        type=["csv"]
    )

if use_default_file:
    data_source = default_file
else:
    data_source = uploaded_file


st.markdown(
    """
    <div class="step-label">Етап 01</div>
    <div class="step-title">Перевірка вхідних даних</div>
    <div class="step-description">
        На цьому етапі перевіряється структура CSV-файлу, наявність потрібних колонок,
        тривалість запису та пропущені значення в основних даних акселерометра.
    </div>
    """,
    unsafe_allow_html=True
)

if data_source is None:
    st.warning("Завантажте CSV-файл або використайте стандартний файл з папки data.")
else:
    try:
        df = load_data(data_source)

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error("У файлі відсутні потрібні колонки:")
            st.write(missing_columns)
        else:
            duration_seconds = df["time_s"].iloc[-1]

            col1, col2, col3 = st.columns(3)

            with col1:
                metric_card("Кількість рядків", f"{len(df)}")

            with col2:
                metric_card("Кількість колонок", f"{df.shape[1]}")

            with col3:
                metric_card("Тривалість запису", f"{duration_seconds:.2f} с")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    '<div class="section-title-small">Основні колонки</div>',
                    unsafe_allow_html=True
                )

                columns_rows = [
                    [col, '<span class="status-badge">знайдено</span>' if col in df.columns else 'не знайдено']
                    for col in required_columns
                ]

                custom_table(["Колонка", "Статус"], columns_rows)

            with col2:
                st.markdown(
                    '<div class="section-title-small">Пропущені значення</div>',
                    unsafe_allow_html=True
                )

                missing_rows = [
                    [col, f'<span class="zero-badge">{df[col].isna().sum()}</span>']
                    for col in required_columns
                ]

                custom_table(["Колонка", "Кількість пропусків"], missing_rows)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            with st.expander("Показати перші 5 рядків основних даних"):
                preview_rows = []

                for _, row in df[required_columns].head().iterrows():
                    preview_rows.append([
                        f"{row['time[us]']}",
                        f"{row['acc_x[mg]']:.3f}",
                        f"{row['acc_y[mg]']:.3f}",
                        f"{row['acc_z[mg]']:.3f}"
                    ])

                custom_table(required_columns, preview_rows)

            green_status("Вхідні дані успішно зчитано і перевірено.")

            # Побудова сирих сигналів акселерометра
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="step-label">Етап 02</div>
                <div class="step-title">Побудова сирих сигналів</div>
                <div class="step-description">
                    На цьому етапі відображаються початкові сигнали акселерометра
                    по трьох осях без попередньої обробки. Це дозволяє побачити
                    вихідну форму сигналів і оцінити їхній загальний характер
                    до центрування та фільтрації.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.sidebar.header("Відображення сирих сигналів")

            show_acc_x = st.sidebar.checkbox("Показати acc_x[mg]", value=True)
            show_acc_y = st.sidebar.checkbox("Показати acc_y[mg]", value=True)
            show_acc_z = st.sidebar.checkbox("Показати acc_z[mg]", value=True)

            col1, col2 = st.columns(2)
            with col1:
                metric_card("Кількість точок", f"{len(df)}")
            with col2:
                metric_card("Тривалість запису", f"{duration_seconds:.2f} с")

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            fig_raw, ax_raw = plt.subplots(figsize=(12, 5))

            if show_acc_x:
                ax_raw.plot(df["time_s"], df["acc_x[mg]"], label="acc_x[mg]", linewidth=1)

            if show_acc_y:
                ax_raw.plot(df["time_s"], df["acc_y[mg]"], label="acc_y[mg]", linewidth=1)

            if show_acc_z:
                ax_raw.plot(df["time_s"], df["acc_z[mg]"], label="acc_z[mg]", linewidth=1)

            ax_raw.set_title("Сирі сигнали акселерометра по трьох осях")
            ax_raw.set_xlabel("Час, с")
            ax_raw.set_ylabel("Прискорення, mg")
            ax_raw.grid(True)
            ax_raw.legend()
            fig_raw.tight_layout()

            st.pyplot(fig_raw)

            green_status(
                "Сирі сигнали містять базове зміщення, пов’язане з положенням датчика та дією гравітації. "
                "На наступному етапі виконується центрування сигналів."
            )

            # Центрування сигналів
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="step-label">Етап 03</div>
                <div class="step-title">Центрування сигналів</div>
                <div class="step-description" style="margin-bottom: 14px;">
                    На цьому етапі з кожної осі акселерометра віднімається її середнє значення.
                    Це дозволяє прибрати базове зміщення, пов’язане з положенням датчика,
                    та перевести сигнали до коливань навколо нуля.
                </div>
                """,
                unsafe_allow_html=True
            )

            centered_df = center_signals(df)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    '<div class="section-title-small" style="margin-top: 8px; margin-bottom: 10px;">Середні значення до центрування</div>',
                    unsafe_allow_html=True
                )

                mean_before_rows = [
                    ["acc_x[mg]", f"{centered_df['acc_x[mg]'].mean():.4f} mg"],
                    ["acc_y[mg]", f"{centered_df['acc_y[mg]'].mean():.4f} mg"],
                    ["acc_z[mg]", f"{centered_df['acc_z[mg]'].mean():.4f} mg"],
                ]

                custom_table(["Сигнал", "Середнє значення"], mean_before_rows)

            with col2:
                st.markdown(
                    '<div class="section-title-small" style="margin-top: 8px; margin-bottom: 10px;">Середні значення після центрування</div>',
                    unsafe_allow_html=True
                )

                mean_after_rows = [
                    ["acc_x_centered", f"{centered_df['acc_x_centered'].mean():.6f} mg"],
                    ["acc_y_centered", f"{centered_df['acc_y_centered'].mean():.6f} mg"],
                    ["acc_z_centered", f"{centered_df['acc_z_centered'].mean():.6f} mg"],
                ]

                custom_table(["Сигнал", "Середнє значення"], mean_after_rows)

            st.sidebar.header("Відображення центрованих сигналів")

            show_centered_x = st.sidebar.checkbox("Показати acc_x_centered", value=True)
            show_centered_y = st.sidebar.checkbox("Показати acc_y_centered", value=True)
            show_centered_z = st.sidebar.checkbox("Показати acc_z_centered", value=True)

            st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

            fig_centered, ax_centered = plt.subplots(figsize=(12, 5))

            if show_centered_x:
                ax_centered.plot(
                    centered_df["time_s"],
                    centered_df["acc_x_centered"],
                    label="acc_x_centered",
                    linewidth=1
                )

            if show_centered_y:
                ax_centered.plot(
                    centered_df["time_s"],
                    centered_df["acc_y_centered"],
                    label="acc_y_centered",
                    linewidth=1
                )

            if show_centered_z:
                ax_centered.plot(
                    centered_df["time_s"],
                    centered_df["acc_z_centered"],
                    label="acc_z_centered",
                    linewidth=1
                )

            ax_centered.set_title("Центровані сигнали акселерометра")
            ax_centered.set_xlabel("Час, с")
            ax_centered.set_ylabel("Відхилення прискорення, mg")
            ax_centered.grid(True)
            ax_centered.legend()
            fig_centered.tight_layout()

            st.pyplot(fig_centered)

            green_status(
                "Після центрування середні значення сигналів стають близькими до нуля. "
                "Це підтверджує, що базове зміщення було прибране."
            )

            # Ресемплінг і фільтрація сигналів
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="step-label">Етап 04</div>
                <div class="step-title">Ресемплінг і фільтрація сигналів</div>
                <div class="step-description">
                    На цьому етапі центровані сигнали переводяться до рівномірної частоти дискретизації
                    та фільтруються у частотному діапазоні, характерному для дихальних коливань.
                    Це допомагає залишити повільні зміни сигналу, пов’язані з диханням,
                    і прибрати зайві високочастотні коливання.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.sidebar.header("Параметри фільтрації")

            target_fs = st.sidebar.number_input(
                "Частота ресемплінгу, Гц",
                min_value=25.0,
                max_value=100.0,
                value=50.0,
                step=25.0
            )

            lowcut = st.sidebar.number_input(
                "Нижня межа фільтра, Гц",
                min_value=0.05,
                max_value=0.50,
                value=0.10,
                step=0.10
            )

            highcut = st.sidebar.number_input(
                "Верхня межа фільтра, Гц",
                min_value=0.30,
                max_value=1.50,
                value=0.70,
                step=0.20
            )

            # Порядок фільтра залишено фіксованим, щоб не перевантажувати інтерфейс.
            if lowcut >= highcut:
                st.error("Нижня межа фільтра має бути меншою за верхню.")
            else:
                filtered_df = filter_signals(
                    centered_df,
                    target_fs,
                    lowcut,
                    highcut,
                    FILTER_ORDER
                )

                st.sidebar.header("Відображення відфільтрованих сигналів")

                show_filtered_x = st.sidebar.checkbox("Показати acc_x_filtered", value=True)
                show_filtered_y = st.sidebar.checkbox("Показати acc_y_filtered", value=True)
                show_filtered_z = st.sidebar.checkbox("Показати acc_z_filtered", value=True)

                top_col1, top_col2 = st.columns(2)

                with top_col1:
                    compact_metric_card("Початкова кількість точок", f"{len(centered_df)}")
                with top_col2:
                    compact_metric_card("Кількість точок після ресемплінгу", f"{len(filtered_df)}")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                bottom_col1, bottom_col2 = st.columns(2)

                with bottom_col1:
                    compact_metric_card("Частота ресемплінгу", f"{target_fs:.2f} Гц")
                with bottom_col2:
                    compact_metric_card("Діапазон фільтрації", f"{lowcut:.2f}–{highcut:.2f} Гц")

                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                fig_filtered, ax_filtered = plt.subplots(figsize=(12, 5))

                if show_filtered_x:
                    ax_filtered.plot(
                        filtered_df["time_s"],
                        filtered_df["acc_x_filtered"],
                        label="acc_x_filtered",
                        linewidth=1.5
                    )

                if show_filtered_y:
                    ax_filtered.plot(
                        filtered_df["time_s"],
                        filtered_df["acc_y_filtered"],
                        label="acc_y_filtered",
                        linewidth=1.5
                    )

                if show_filtered_z:
                    ax_filtered.plot(
                        filtered_df["time_s"],
                        filtered_df["acc_z_filtered"],
                        label="acc_z_filtered",
                        linewidth=1.5
                    )

                ax_filtered.set_title("Відфільтровані сигнали акселерометра")
                ax_filtered.set_xlabel("Час, с")
                ax_filtered.set_ylabel("Амплітуда після фільтрації, mg")
                ax_filtered.grid(True)
                ax_filtered.legend()
                fig_filtered.tight_layout()

                st.pyplot(fig_filtered)

                green_status(
                    "Після фільтрації залишаються повільні коливання, які можуть відповідати "
                    "дихальним рухам. Ці сигнали використовуються на наступних етапах для пошуку піків."
                )

                # Базовий пошук піків
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="step-label">Етап 05</div>
                    <div class="step-title">Базовий пошук піків</div>
                    <div class="step-description">
                        На цьому етапі на вибраному відфільтрованому сигналі визначаються локальні максимуми.
                        Саме ці піки надалі використовуються як дихальні цикли для розрахунку частоти дихання.
                        Параметри min_distance та prominence можна змінювати в бічній панелі.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.sidebar.header("Параметри пошуку піків")

                signal_column = st.sidebar.selectbox(
                    "Сигнал для аналізу",
                    ["acc_y_filtered", "acc_x_filtered", "acc_z_filtered"],
                    index=0
                )

                min_distance_seconds = st.sidebar.select_slider(
                    "Мінімальна відстань між піками, с",
                    options=[2.0, 4.0, 5.0, 6.0, 8.0],
                    value=2.0
                )

                prominence_value = st.sidebar.select_slider(
                    "Prominence",
                    options=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
                    value=2.0
                )

                time = filtered_df["time_s"]
                signal = filtered_df[signal_column]

                peaks, properties, distance_samples = detect_peaks(
                    signal,
                    target_fs,
                    min_distance_seconds,
                    prominence_value
                )

                peaks_count = len(peaks)

                peaks_df = pd.DataFrame({
                    "peak_index": peaks,
                    "time_s": time.iloc[peaks].values,
                    "amplitude": signal.iloc[peaks].values,
                    "prominence": properties["prominences"]
                })

                top_col1, top_col2 = st.columns(2)

                with top_col1:
                    compact_metric_card("Сигнал для аналізу", signal_column)
                with top_col2:
                    compact_metric_card("Кількість знайдених піків", f"{peaks_count}")

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                bottom_col1, bottom_col2, bottom_col3 = st.columns(3)

                with bottom_col1:
                    compact_metric_card("Мінімальна відстань", f"{min_distance_seconds:.1f} с")
                with bottom_col2:
                    compact_metric_card("Відстань у точках", f"{distance_samples}")
                with bottom_col3:
                    compact_metric_card("Prominence", f"{prominence_value:.1f}")

                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                fig_peaks, ax_peaks = plt.subplots(figsize=(12, 5))

                ax_peaks.plot(
                    time,
                    signal,
                    label=signal_column,
                    linewidth=1.5
                )

                ax_peaks.scatter(
                    time.iloc[peaks],
                    signal.iloc[peaks],
                    color="red",
                    label="Знайдені піки",
                    zorder=3
                )

                ax_peaks.set_title("Пошук піків на відфільтрованому сигналі")
                ax_peaks.set_xlabel("Час, с")
                ax_peaks.set_ylabel("Амплітуда після фільтрації, mg")
                ax_peaks.grid(True)
                ax_peaks.legend()
                fig_peaks.tight_layout()

                st.pyplot(fig_peaks)

                with st.expander("Показати таблицю знайдених піків"):
                    peaks_rows = []

                    for _, row in peaks_df.iterrows():
                        peaks_rows.append([
                            f"{int(row['peak_index'])}",
                            f"{row['time_s']:.2f}",
                            f"{row['amplitude']:.4f}",
                            f"{row['prominence']:.4f}"
                        ])

                    custom_table(
                        ["peak_index", "time_s", "amplitude", "prominence"],
                        peaks_rows
                    )

                green_status(
                    "Червоні точки на графіку позначають знайдені піки сигналу. "
                    "Зміна параметрів min_distance та prominence впливає на кількість піків, "
                    "які буде враховано алгоритмом."
                )

                # Розрахунок частоти дихання
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="step-label">Етап 06</div>
                    <div class="step-title">Розрахунок частоти дихання</div>
                    <div class="step-description">
                        На цьому етапі кількість знайдених піків переводиться у частоту дихання.
                        Один знайдений пік приймається як один дихальний цикл, а результат
                        розраховується у диханнях за хвилину.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                start_time = filtered_df["time_s"].iloc[0]
                end_time = filtered_df["time_s"].iloc[-1]

                duration_seconds = end_time - start_time
                duration_minutes = duration_seconds / 60

                if duration_minutes > 0:
                    breathing_rate = peaks_count / duration_minutes
                else:
                    breathing_rate = 0

                rate_result_df = pd.DataFrame({
                    "signal": [signal_column],
                    "min_distance_seconds": [min_distance_seconds],
                    "prominence": [prominence_value],
                    "duration_seconds": [round(duration_seconds, 2)],
                    "peaks_count": [peaks_count],
                    "breathing_rate_bpm": [round(breathing_rate, 2)]
                })

                col1, col2, col3 = st.columns(3)

                with col1:
                    compact_metric_card("Тривалість запису", f"{duration_seconds:.2f} с")
                with col2:
                    compact_metric_card("Кількість піків", f"{peaks_count}")
                with col3:
                    compact_metric_card("Частота дихання", f"{breathing_rate:.2f} дих./хв")

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-title-small">Результат розрахунку</div>',
                    unsafe_allow_html=True
                )

                dataframe_to_custom_table(
                    rate_result_df,
                    columns=[
                        "signal",
                        "min_distance_seconds",
                        "prominence",
                        "duration_seconds",
                        "peaks_count",
                        "breathing_rate_bpm"
                    ],
                    formatters={
                        "min_distance_seconds": lambda v: f"{float(v):.1f}",
                        "prominence": lambda v: f"{float(v):.1f}",
                        "duration_seconds": lambda v: f"{float(v):.2f}",
                        "breathing_rate_bpm": lambda v: f"{float(v):.2f}",
                    }
                )

                green_status(
                    "Частота дихання розраховується як відношення кількості знайдених піків "
                    "до тривалості запису у хвилинах. Якщо змінити сигнал або параметри пошуку піків, "
                    "результат автоматично перерахується."
                )

                # Порівняння параметрів пошуку піків
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="step-label">Етап 07</div>
                    <div class="step-title">Порівняння параметрів пошуку піків</div>
                    <div class="step-description">
                        На цьому етапі перевіряється, як зміна параметрів пошуку піків впливає
                        на кількість знайдених дихальних циклів і розраховану частоту дихання.
                        Для наочності на графіку змінюється тільки один параметр, а інший
                        залишається фіксованим.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.sidebar.header("Параметри порівняння")

                comparison_mode = st.sidebar.selectbox(
                    "Що показати на графіку",
                    [
                        "Вплив prominence",
                        "Вплив min_distance"
                    ],
                    index=0
                )

                signals_for_comparison = ["acc_x_filtered", "acc_y_filtered", "acc_z_filtered"]

                if comparison_mode == "Вплив prominence":
                    fixed_min_distance = 2.0
                    comparison_values = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
                    params_comparison_df = compare_peak_params(
                        filtered_df,
                        target_fs,
                        [fixed_min_distance],
                        comparison_values
                    )

                    plot_df = params_comparison_df[[
                        "signal",
                        "prominence",
                        "peaks_count",
                        "breathing_rate_bpm"
                    ]].copy()

                    x_column = "prominence"
                    x_label = "Prominence"
                    graph_title = f"Вплив prominence на частоту дихання при min_distance = {fixed_min_distance:.1f} с"
                    fixed_parameter_text = f"min_distance = {fixed_min_distance:.1f} с"

                else:
                    fixed_prominence = 2.0
                    comparison_values = [2.0, 4.0, 5.0, 6.0, 8.0]
                    params_comparison_df = compare_peak_params(
                        filtered_df,
                        target_fs,
                        comparison_values,
                        [fixed_prominence]
                    )

                    plot_df = params_comparison_df[[
                        "signal",
                        "min_distance_seconds",
                        "peaks_count",
                        "breathing_rate_bpm"
                    ]].copy()

                    x_column = "min_distance_seconds"
                    x_label = "Min distance, с"
                    graph_title = f"Вплив min_distance на частоту дихання при prominence = {fixed_prominence:.1f}"
                    fixed_parameter_text = f"prominence = {fixed_prominence:.1f}"

                col1, col2 = st.columns(2)

                with col1:
                    compact_metric_card("Режим порівняння", comparison_mode)
                with col2:
                    compact_metric_card("Фіксований параметр", fixed_parameter_text)

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                fig_params, ax_params = plt.subplots(figsize=(12, 5))

                line_styles = {
                    "acc_x_filtered": {"marker": "o", "linestyle": "-", "label": "acc_x_filtered"},
                    "acc_y_filtered": {"marker": "o", "linestyle": "--", "label": "acc_y_filtered"},
                    "acc_z_filtered": {"marker": "o", "linestyle": "-.", "label": "acc_z_filtered"},
                }

                for signal_name in signals_for_comparison:
                    signal_data = plot_df[plot_df["signal"] == signal_name]

                    ax_params.plot(
                        signal_data[x_column],
                        signal_data["breathing_rate_bpm"],
                        marker=line_styles[signal_name]["marker"],
                        linestyle=line_styles[signal_name]["linestyle"],
                        linewidth=2,
                        markersize=8,
                        label=line_styles[signal_name]["label"]
                    )

                ax_params.set_title(graph_title)
                ax_params.set_xlabel(x_label)
                ax_params.set_ylabel("Частота дихання, дих./хв")
                ax_params.grid(True)
                ax_params.legend()
                fig_params.tight_layout()

                st.pyplot(fig_params)

                with st.expander("Показати таблицю розрахунків для графіка"):
                    if comparison_mode == "Вплив prominence":
                        dataframe_to_custom_table(
                            plot_df,
                            columns=[
                                "signal",
                                "prominence",
                                "peaks_count",
                                "breathing_rate_bpm"
                            ],
                            formatters={
                                "prominence": lambda v: f"{float(v):.1f}",
                                "breathing_rate_bpm": lambda v: f"{float(v):.2f}",
                            }
                        )
                    else:
                        dataframe_to_custom_table(
                            plot_df,
                            columns=[
                                "signal",
                                "min_distance_seconds",
                                "peaks_count",
                                "breathing_rate_bpm"
                            ],
                            formatters={
                                "min_distance_seconds": lambda v: f"{float(v):.1f}",
                                "breathing_rate_bpm": lambda v: f"{float(v):.2f}",
                            }
                        )

                green_status(
                    "На цьому графіку змінюється лише один параметр пошуку піків. "
                    "Якщо збільшується prominence, алгоритм залишає тільки більш виразні піки. "
                    "Якщо збільшується min_distance, алгоритм не дозволяє рахувати піки, "
                    "які розташовані занадто близько один до одного."
                )

                # Фінальний результат для основного запису
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="step-label">Етап 08</div>
                    <div class="step-title">Фінальний результат для основного запису</div>
                    <div class="step-description">
                        На цьому етапі фіксується підсумковий результат для основного запису.
                        Для фінального розрахунку використовується сигнал acc_y_filtered
                        з параметрами min_distance = 2,0 с та prominence = 2,0.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                final_signal_column = "acc_y_filtered"
                final_min_distance = 2.0
                final_prominence = 2.0

                final_time = filtered_df["time_s"]
                final_signal = filtered_df[final_signal_column]

                final_distance_samples = int(final_min_distance * target_fs)

                final_peaks, final_properties = find_peaks(
                    final_signal,
                    distance=final_distance_samples,
                    prominence=final_prominence
                )

                final_peaks_count = len(final_peaks)
                final_duration_seconds = filtered_df["time_s"].iloc[-1] - filtered_df["time_s"].iloc[0]
                final_duration_minutes = final_duration_seconds / 60

                if final_duration_minutes > 0:
                    final_breathing_rate = final_peaks_count / final_duration_minutes
                else:
                    final_breathing_rate = 0

                final_peaks_df = pd.DataFrame({
                    "peak_index": final_peaks,
                    "time_s": final_time.iloc[final_peaks].values,
                    "amplitude": final_signal.iloc[final_peaks].values,
                    "prominence": final_properties["prominences"]
                })

                col1, col2, col3 = st.columns(3)

                with col1:
                    compact_metric_card("Фінальна частота дихання", f"{final_breathing_rate:.2f} дих./хв")
                with col2:
                    compact_metric_card("Кількість піків", f"{final_peaks_count}")
                with col3:
                    compact_metric_card("Обраний сигнал", final_signal_column)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                col4, col5, col6 = st.columns(3)

                with col4:
                    compact_metric_card("Тривалість запису", f"{final_duration_seconds:.2f} с")
                with col5:
                    compact_metric_card("Мінімальна відстань", f"{final_min_distance:.1f} с")
                with col6:
                    compact_metric_card("Prominence", f"{final_prominence:.1f}")

                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                fig_final, ax_final = plt.subplots(figsize=(12, 5))

                ax_final.plot(
                    final_time,
                    final_signal,
                    label=final_signal_column,
                    linewidth=1.5
                )

                ax_final.scatter(
                    final_time.iloc[final_peaks],
                    final_signal.iloc[final_peaks],
                    color="red",
                    label="Знайдені піки",
                    zorder=3
                )

                ax_final.set_title("Результат пошуку піків для основного запису")
                ax_final.set_xlabel("Час, с")
                ax_final.set_ylabel("Амплітуда після фільтрації, mg")
                ax_final.grid(True)
                ax_final.legend()
                fig_final.tight_layout()

                st.pyplot(fig_final)

                with st.expander("Показати координати фінальних піків"):
                    dataframe_to_custom_table(
                        final_peaks_df,
                        columns=[
                            "peak_index",
                            "time_s",
                            "amplitude",
                            "prominence"
                        ],
                        formatters={
                            "peak_index": lambda v: f"{int(v)}",
                            "time_s": lambda v: f"{float(v):.2f}",
                            "amplitude": lambda v: f"{float(v):.4f}",
                            "prominence": lambda v: f"{float(v):.4f}",
                        }
                    )

                green_status(
                    f"Фінальний результат для основного запису: знайдено "
                    f"{final_peaks_count} піків, що відповідає частоті дихання "
                    f"{final_breathing_rate:.2f} дих./хв."
                )

                # Порівняння різних датасетів
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="step-label">Етап 09</div>
                    <div class="step-title">Порівняння різних датасетів</div>
                    <div class="step-description">
                        На цьому етапі той самий pipeline обробки застосовується до трьох
                        додаткових акселерометричних записів: нормального дихання,
                        повільного дихання та запису з кашлем або артефактами.
                        Це дозволяє перевірити роботу алгоритму не лише на одному файлі,
                        а й на різних типах сигналів.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                datasets_df, processed_signals = compare_external_datasets(
                    target_fs=target_fs,
                    lowcut=lowcut,
                    highcut=highcut,
                    order=FILTER_ORDER,
                    min_distance_seconds=2.0
                )

                if datasets_df.empty:
                    st.warning(
                        "Додаткові датасети не знайдено. Перевірте наявність файлів "
                        "acc_normal.csv, acc_slow.csv та acc_cough.csv у папці data."
                    )
                else:
                    col1, col2, col3 = st.columns(3)

                    normal_row = datasets_df[datasets_df["dataset"] == "Нормальне дихання"].iloc[0]
                    slow_row = datasets_df[datasets_df["dataset"] == "Повільне дихання"].iloc[0]
                    cough_row = datasets_df[datasets_df["dataset"] == "Кашель / артефакти"].iloc[0]

                    with col1:
                        compact_metric_card(
                            "Нормальне дихання",
                            f"{normal_row['breathing_rate_bpm']:.2f} дих./хв"
                        )

                    with col2:
                        compact_metric_card(
                            "Повільне дихання",
                            f"{slow_row['breathing_rate_bpm']:.2f} дих./хв"
                        )

                    with col3:
                        compact_metric_card(
                            "Кашель / артефакти",
                            f"{cough_row['breathing_rate_bpm']:.2f} дих./хв"
                        )

                    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                    st.markdown(
                        '<div class="section-title-small">Таблиця порівняння датасетів</div>',
                        unsafe_allow_html=True
                    )

                    dataframe_to_custom_table(
                        datasets_df,
                        columns=[
                            "dataset",
                            "duration_seconds",
                            "original_fs_hz",
                            "prominence",
                            "peaks_count",
                            "breathing_rate_bpm",
                            "comment"
                        ],
                        formatters={
                            "duration_seconds": lambda v: f"{float(v):.2f}",
                            "original_fs_hz": lambda v: f"{float(v):.2f}",
                            "prominence": lambda v: f"{float(v):.4f}",
                            "breathing_rate_bpm": lambda v: f"{float(v):.2f}",
                        }
                    )

                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                    fig_datasets, ax_datasets = plt.subplots(figsize=(12, 5))

                    bars = ax_datasets.bar(
                        datasets_df["dataset"],
                        datasets_df["breathing_rate_bpm"]
                    )

                    ax_datasets.set_title("Порівняння частоти дихання для різних датасетів")
                    ax_datasets.set_xlabel("Датасет")
                    ax_datasets.set_ylabel("Частота дихання, дих./хв")
                    ax_datasets.grid(axis="y")

                    for bar in bars:
                        height = bar.get_height()

                        ax_datasets.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.3,
                            f"{height:.2f}",
                            ha="center",
                            va="bottom"
                        )

                    fig_datasets.tight_layout()
                    st.pyplot(fig_datasets)

                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

                    fragment_seconds = 30

                    fig_overview, axes = plt.subplots(
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
                        peaks_dataset = item["peaks"]
                        breathing_rate_dataset = item["breathing_rate"]
                        dataset_name = item["dataset"]

                        fragment_mask = time_values <= fragment_seconds

                        fragment_time = time_values[fragment_mask]
                        fragment_signal = filtered_signal[fragment_mask]

                        fragment_peak_indices = [
                            peak for peak in peaks_dataset
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
                            f"{dataset_name}: {breathing_rate_dataset:.2f} дих./хв",
                            fontsize=12
                        )

                        ax.set_ylabel("Амплітуда")
                        ax.grid(True)
                        ax.legend(loc="upper right")

                    axes[-1].set_xlabel("Час, с")

                    fig_overview.suptitle(
                        "Фрагменти відфільтрованих сигналів для різних датасетів",
                        fontsize=14
                    )

                    fig_overview.tight_layout(rect=[0, 0, 1, 0.95])
                    st.pyplot(fig_overview)

                    green_status(
                        "Порівняння показує, що один і той самий pipeline обробки можна застосувати "
                        "до різних акселерометричних записів. Для повільного дихання частота є нижчою, "
                        "а запис із кашлем або артефактами містить різкі коливання, які можуть впливати "
                        "на кількість знайдених піків."
                    )

    except Exception as error:
        st.error("Під час зчитування або обробки файлу виникла помилка.")
        st.write(error)