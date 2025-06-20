import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import os
import gc
import time
import joblib

# --- Cấu hình ---
DATA_PATH = "E:\\Demo\\TIMESERIES\\Data1.csv"
MODEL_DIR = "models"
MAX_HOUSEHOLDS = 5
DATE_MIN = datetime(2011, 12, 1)
DATE_MAX = datetime(2014, 2, 28)

# --- Thiết bị ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Các bước dự báo multi-step ---
forecast_steps_map = {
    "1 giờ": 2,
    "1 ngày": 48
}

# --- Độ dài chuỗi đầu vào tương ứng ---
seq_len_map = {
    "1 giờ": 48,
    "1 ngày": 336
}

# --- Mô hình LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, output_dim=48):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# --- Xử lý chuỗi 0 dài ---
def clean_long_zero_sequences(series, threshold=6):
    zero_mask = (series == 0)
    group = (zero_mask != zero_mask.shift()).cumsum()
    counts = zero_mask.groupby(group).transform("sum")
    to_nan = (zero_mask & (counts >= threshold))
    series_cleaned = series.copy()
    series_cleaned[to_nan] = np.nan
    return series_cleaned.interpolate(method="linear").ffill().bfill()

# --- Load dữ liệu ---
@st.cache_data(show_spinner=False)
def load_full_data(path):
    chunks = pd.read_csv(path, sep=';', engine="c", chunksize=95_000, on_bad_lines='skip')
    df_list = []
    for chunk in chunks:
        chunk.columns = chunk.columns.str.strip()
        if "KWH/hh (per half hour)" in chunk.columns:
            chunk["KWH/hh (per half hour)"] = pd.to_numeric(
                chunk["KWH/hh (per half hour)"].astype(str).str.replace(",", "."), errors='coerce')
        df_list.append(chunk)

    df = pd.concat(df_list, ignore_index=True)
    del df_list, chunks
    gc.collect()

    df.dropna(subset=["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour)"], inplace=True)
    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors='coerce')
    df.dropna(subset=["DateTime"], inplace=True)
    df.set_index("DateTime", inplace=True)

    return df[(df.index >= DATE_MIN) & (df.index <= DATE_MAX)]

@st.cache_data(show_spinner=False)
def load_available_households(df):
    return sorted(df["LCLid"].unique())

# --- Xử lý dữ liệu hộ ---
def get_household_data(df, household_id, start_date, end_date):
    df_house = df[df["LCLid"] == household_id]
    df_house = df_house[(df_house.index >= pd.to_datetime(start_date)) & (df_house.index <= pd.to_datetime(end_date))]
    if df_house.empty:
        return None
    ts = df_house["KWH/hh (per half hour)"].resample("30min").mean().ffill()
    ts = ts[ts >= 0]  # Bỏ giá trị âm
    ts = clean_long_zero_sequences(ts)  # Làm sạch chuỗi 0 dài
    ts = ts.clip(upper=ts.quantile(0.995))  # Loại outlier
    return ts

# --- Chuẩn bị chuỗi đầu vào ---
def prepare_sequence(series, seq_len):
    values = series.values.reshape(-1, 1)
    if len(values) < seq_len:
        return None, None
    return values[-seq_len:], None

# --- Load mô hình + scaler ---
def load_model_and_scaler(household_id, label):
    folder_name = f"{household_id}_{label}"
    folder_path = os.path.join(MODEL_DIR, folder_name)
    model_path = os.path.join(folder_path, "final_model.pt")
    scaler_path = os.path.join(folder_path, "scaler.save")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    forecast_steps = forecast_steps_map[label]
    model = LSTMModel(output_dim=forecast_steps).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)

    return model, scaler

# --- Dự báo ---
def forecast_multi_step(model, input_seq, scaler, seq_len):
    scaled = scaler.transform(input_seq)
    input_tensor = torch.tensor(scaled.reshape(1, seq_len, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy().reshape(-1, 1)
    return scaler.inverse_transform(output).flatten()

# --- Giao diện ---
st.set_page_config(page_title="Dự báo điện", layout="wide")
st.title("DỰ BÁO ĐIỆN NĂNG TIÊU THỤ")

with st.spinner("📦 Đang tải dữ liệu..."):
    full_df = load_full_data(DATA_PATH)
    households = load_available_households(full_df)

selected_households = st.multiselect("Chọn hộ gia đình", households, max_selections=MAX_HOUSEHOLDS)

start_date = st.date_input("Từ ngày", datetime(2011, 12, 1), min_value=DATE_MIN, max_value=DATE_MAX)
end_date = st.date_input("Đến ngày", datetime(2014, 2, 28), min_value=DATE_MIN, max_value=DATE_MAX)

if start_date > end_date:
    st.warning("❗ Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

forecast_label = st.selectbox("Khoảng thời gian dự báo", list(forecast_steps_map.keys()))

if st.button("Dự báo"):
    if not selected_households:
        st.warning("Vui lòng chọn ít nhất 1 hộ.")
        st.stop()

    seq_len = seq_len_map[forecast_label]

    for hid in selected_households:
        st.subheader(f"{hid}")
        start = time.time()

        ts = get_household_data(full_df, hid, start_date, end_date)
        if ts is None or len(ts) < seq_len:
            st.warning("Không đủ dữ liệu.")
            continue

        input_seq, _ = prepare_sequence(ts, seq_len)
        if input_seq is None:
            st.warning("Chuỗi đầu vào không hợp lệ.")
            continue

        model, scaler = load_model_and_scaler(hid, forecast_label)
        if model is None or scaler is None:
            st.warning("Không tìm thấy mô hình phù hợp.")
            continue

        preds = forecast_multi_step(model, input_seq, scaler, seq_len)
        future_index = [ts.index[-1] + timedelta(minutes=30 * (i + 1)) for i in range(len(preds))]

        forecast_df = pd.DataFrame({
            "Thời gian": future_index,
            "Dự báo (kWh)": preds
        }).set_index("Thời gian")

        st.line_chart(forecast_df)

        end = time.time()
        st.success(f"Dự báo hoàn thành trong {end - start:.2f} giây.")



# cd /d e:\Demo\TIMESERIES
# streamlit run GD_demo.py

