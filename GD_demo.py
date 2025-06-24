import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
import zipfile
import os
import gc
import time
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download

# ======================= DOWNLOAD DATA FROM HUGGING FACE =======================

# Download datafull.zip
if not os.path.exists("datafull.zip"):
    hf_hub_download(repo_id="An272/dudoandiennang", filename="datafull.zip",repo_type="dataset" local_dir=".")
    with zipfile.ZipFile("datafull.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Download model.zip
if not os.path.exists("model.zip"):
    hf_hub_download(repo_id="An272/dudoandiennang", filename="model.zip",repo_type="dataset", local_dir=".")

if not os.path.exists("model"):
    os.makedirs("model")

with zipfile.ZipFile("model.zip", 'r') as zip_ref:
    zip_ref.extractall("model")


# ======================= CONFIGURATION =======================
DATA_PATH = DATA_PATH = "E:/Demo/datafull/CC_LCL-FullData.csv"

MODEL_DIR = "model"
MAX_HOUSEHOLDS = 5
DATE_MIN = datetime(2011, 12, 1)
DATE_MAX = datetime(2014, 2, 28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================= MODEL DEFINITION =======================
FORECAST_STEPS = 48  # 1 ngày tiếp theo (24h * 2)
SEQ_LEN = 336        # 7 ngày quan sát (7 * 24h * 2)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, output_dim=FORECAST_STEPS):
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
    chunks = pd.read_csv(path, sep=',', engine="python", chunksize=95000, on_bad_lines='skip')
    df_list = []
    for chunk in chunks:
        chunk.columns = chunk.columns.str.strip()
        if not set(["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour)"]).issubset(chunk.columns):
            continue  # Bỏ qua chunk nếu thiếu cột
        chunk = chunk[["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour)"]]
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

# Load tất cả models vào RAM trước
all_models = load_all_models()

# ======================= STREAMLIT APP =======================
st.set_page_config(page_title="Dự báo điện năng 12 giờ", layout="wide")
st.title("🔋 DỰ BÁO ĐIỆN NĂNG TIÊU THỤ")

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

        if hid not in all_models:
            st.warning("Không tìm thấy mô hình đã huấn luyện cho hộ này.")
            continue

        model = LSTMModel().to(device)
        model.load_state_dict(all_models[hid])
        model.eval()

        input_tensor = torch.tensor(input_seq.reshape(1, SEQ_LEN, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(input_tensor).cpu().numpy().reshape(-1, 1)
        preds = output.flatten()

        future_index = [ts.index[-1] + timedelta(minutes=30 * (i + 1)) for i in range(len(preds))]
        forecast_df = pd.DataFrame({"Thời gian": future_index, "Dự báo (kWh)": preds}).set_index("Thời gian")
        st.line_chart(forecast_df)

        end = time.time()
        st.success(f"Dự báo hoàn thành trong {end - start:.2f} giây.")
