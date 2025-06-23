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
import gdown
import zipfile

# ======================= DOWNLOAD DATA FROM GOOGLE DRIVE =======================

# Download data zip
if not os.path.exists("datafull.zip"):
    file_id = "1JTKRdO-24v4oUQXyPhRYGtlX8b1yaCZ5"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, "datafull.zip", quiet=False)

# Download model zip
if not os.path.exists("model.zip"):
    file_id = "1ZuC_LHycA0gcAHJ5D6XB8yLzT8ouNT87"
    gdown.download(id=file_id, output="model.zip", quiet=False)
    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall("model")  # giải nén vào thư mục model

# ======================= CONFIGURATION =======================
DATA_PATH = "DataSample.csv"
MODEL_DIR = "model"
MAX_HOUSEHOLDS = 5
DATE_MIN = datetime(2011, 12, 1)
DATE_MAX = datetime(2014, 2, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= MODEL DEFINITION =======================
FORECAST_STEPS = 1  # 1 ngày tiếp theo
SEQ_LEN = 336        # 7 ngày quan sát

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

# ======================= DATA PREPROCESSING =======================
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

# ======================= LOAD MODELS FROM MULTI-PT =======================
@st.cache_data(show_spinner=False)
def load_all_models():
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    full_model_dict = {}
    for file in model_files:
        file_path = os.path.join(MODEL_DIR, file)
        state_dict_all = torch.load(file_path, map_location=device)
        for hid, state_dict in state_dict_all.items():
            full_model_dict[hid] = state_dict
    return full_model_dict

all_models = load_all_models()

# ======================= STREAMLIT APP =======================
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

if st.button("Dự báo ngày tiếp theo"):
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

        # Load model cho đúng household id
        if hid not in all_models:
            st.warning("Không tìm thấy mô hình đã huấn luyện.")
            continue

        model = LSTMModel(output_dim=FORECAST_STEPS).to(device)
        model.load_state_dict(all_models[hid])
        model.eval()

        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, "scaler", f"{hid}_scaler.save")
        if not os.path.exists(scaler_path):
            st.warning("Không tìm thấy scaler.")
            continue
        scaler = joblib.load(scaler_path)

        # Dự báo
        scaled = scaler.transform(input_seq)
        input_tensor = torch.tensor(scaled.reshape(1, SEQ_LEN, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(input_tensor).cpu().numpy().reshape(-1, 1)
        preds = scaler.inverse_transform(output).flatten()

        future_index = [ts.index[-1] + timedelta(minutes=30 * (i + 1)) for i in range(len(preds))]
        forecast_df = pd.DataFrame({"Thời gian": future_index, "Dự báo (kWh)": preds}).set_index("Thời gian")
        st.line_chart(forecast_df)
        end = time.time()
        st.success(f"Dự báo hoàn thành trong {end - start:.2f} giây.")