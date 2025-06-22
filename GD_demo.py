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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tham số cố định theo model mới ---
FORECAST_STEPS = 10  # 5 giờ tiếp theo
SEQ_LEN = 48        # 1 ngày quan sát

# --- Mô hình LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, output_dim=24):
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

@st.cache_data(show_spinner=False)
def load_full_data(path):
    use_cols = ["LCLid", "DateTime", "KWH/hh (per half hour)"]
    chunks = pd.read_csv(path, sep=';', usecols=use_cols, engine="c", chunksize=95_000, on_bad_lines='skip')
    df_list = []
    for chunk in chunks:
        chunk.columns = chunk.columns.str.strip()
        chunk["KWH/hh (per half hour)"] = pd.to_numeric(
            chunk["KWH/hh (per half hour)"].astype(str).str.replace(",", "."), errors='coerce')
        chunk["DateTime"] = pd.to_datetime(chunk["DateTime"], dayfirst=True, errors='coerce')
        chunk.dropna(subset=["LCLid", "DateTime", "KWH/hh (per half hour)"], inplace=True)
        chunk = chunk[(chunk["DateTime"] >= DATE_MIN) & (chunk["DateTime"] <= DATE_MAX)]
        df_list.append(chunk)
    df = pd.concat(df_list, ignore_index=True)
    df.set_index("DateTime", inplace=True)
    del df_list
    gc.collect()
    return df

@st.cache_data(show_spinner=False)
def load_available_households(df):
    return sorted(df["LCLid"].unique())

def get_household_data(df, household_id, start_date, end_date):
    df_house = df[df["LCLid"] == household_id]
    df_house = df_house[(df_house.index >= pd.to_datetime(start_date)) & (df_house.index <= pd.to_datetime(end_date))]
    if df_house.empty:
        return None
    ts = df_house["KWH/hh (per half hour)"].resample("30min").mean().ffill()
    ts = ts[ts >= 0]
    ts = clean_long_zero_sequences(ts)
    ts = ts.clip(upper=ts.quantile(0.995))
    return ts

def prepare_sequence(series):
    values = series.values.reshape(-1, 1)
    if len(values) < SEQ_LEN:
        return None
    return values[-SEQ_LEN:]

def load_model_and_scaler(household_id):
    folder_name = f"{household_id}_12h"
    folder_path = os.path.join(MODEL_DIR, folder_name)
    model_path = os.path.join(folder_path, "final_model.pt")
    scaler_path = os.path.join(folder_path, "scaler.save")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = LSTMModel(output_dim=FORECAST_STEPS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)

    return model, scaler

def forecast_multi_step(model, input_seq, scaler):
    scaled = scaler.transform(input_seq)
    input_tensor = torch.tensor(scaled.reshape(1, SEQ_LEN, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy().reshape(-1, 1)
    return scaler.inverse_transform(output).flatten()

# --- Giao diện chính ---
st.set_page_config(page_title="Dự báo điện năng 12 giờ", layout="wide")
st.title("🔋 DỰ BÁO ĐIỆN NĂNG TIÊU THỤ ")

with st.spinner("📦 Đang tải dữ liệu..."):
    full_df = load_full_data(DATA_PATH)
    households = load_available_households(full_df)

selected_households = st.multiselect("Chọn hộ gia đình", households, max_selections=MAX_HOUSEHOLDS)

start_date = st.date_input("Từ ngày", datetime(2011, 12, 1), min_value=DATE_MIN, max_value=DATE_MAX)
end_date = st.date_input("Đến ngày", datetime(2014, 2, 28), min_value=DATE_MIN, max_value=DATE_MAX)

if start_date > end_date:
    st.warning("❗ Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

if st.button("Dự báo 12 giờ tiếp theo"):
    if not selected_households:
        st.warning("Vui lòng chọn ít nhất 1 hộ.")
        st.stop()

    for hid in selected_households:
        st.subheader(f"Hộ: {hid}")
        start = time.time()

        ts = get_household_data(full_df, hid, start_date, end_date)
        if ts is None or len(ts) < SEQ_LEN:
            st.warning("Không đủ dữ liệu để dự báo.")
            continue

        input_seq = prepare_sequence(ts)
        if input_seq is None:
            st.warning("Chuỗi đầu vào không hợp lệ.")
            continue

        model, scaler = load_model_and_scaler(hid)
        if model is None or scaler is None:
            st.warning("Không tìm thấy mô hình đã huấn luyện.")
            continue

        preds = forecast_multi_step(model, input_seq, scaler)
        future_index = [ts.index[-1] + timedelta(minutes=30 * (i + 1)) for i in range(len(preds))]

        forecast_df = pd.DataFrame({
            "Thời gian": future_index,
            "Dự báo (kWh)": preds
        }).set_index("Thời gian")

        st.line_chart(forecast_df)
        end = time.time()
        st.success(f"Dự báo hoàn thành trong {end - start:.2f} giây.")
