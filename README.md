DỰ BÁO SIÊU CHI TIẾT NHU CẦU ĐIỆN NĂNG
📌 Mô tả đề tài
Trong thời đại chuyển đổi số và đô thị thông minh, nhu cầu sử dụng điện năng ngày càng tăng và biến động mạnh mẽ theo thời gian. Đặc biệt tại các đô thị lớn như London, sự đa dạng trong hành vi tiêu dùng của các hộ gia đình, kết hợp với ảnh hưởng của thời tiết, ngày lễ và các hoạt động xã hội, tạo ra những thách thức không nhỏ trong việc quản lý và phân phối điện năng một cách hiệu quả. Trước tình hình đó, khả năng dự báo chính xác nhu cầu điện năng ở mức độ chi tiết cao trở nên cần thiết hơn bao giờ hết. Đề tài này tập trung vào bài toán dự báo điện năng tiêu thụ trong tương lai gần tại các hộ gia đình ở London, dựa trên dữ liệu được thu thập từ hệ thống smart meter. Dữ liệu có độ phân giải cao (mỗi 30 phút) và bao phủ hơn 5,500 hộ gia đình trong khoảng thời gian từ cuối năm 2011 đến đầu năm 2014, với tổng dung lượng sau giải nén lên đến hơn 10GB. 

📚 Dữ liệu sử dụng
📥 Nguồn dữ liệu: [Time Series H1N1 Tweets (Kaggle)](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
💾 Bao gồm các đặc trưng như: LCLid,StdorToU, DateTime, KWH/h.

🔄 Được xử lý thống nhất qua:
Xử lý datetime, chuẩn hóa (StandardScaler, MinMaxScaler)
Đọc dữ liệu từng hộ theo từng khối (Chunk) 
Phát hiện các chuỗi giá trị 0 liên tiếp có độ dài ≥ threshold (mặc định là 6). Những chuỗi này được thay bằng NaN, sau đó nội suy (interpolate) để tránh làm sai lệch mô hình.
Sau đó, nhóm lọc ra 5 hộ có tổng mức tiêu thụ cao nhất để làm ví dụ cho quá trình huấn luyện và đánh giá mô hình. Việc chọn top hộ tiêu thụ cao giúp đảm bảo dữ liệu đầu vào đủ dài và có nhiều biến động, giúp mô hình học được nhiều mẫu thời gian hơn.

🧠 Mô hình triển khai
Ba mô hình học sâu được so sánh:
1. LSTM
LSTM có khả năng tự học “nhớ” hay “quên” thông tin qua thời gian. Do đó, LSTM giải quyết tốt hơn vấn đề mất trí nhớ thông thường của RNN, cho phép mạng duy trì ảnh hưởng của các sự kiện xảy ra từ xa trong dữ liệu chuỗi. Nói cách khác, LSTM học được những đặc điểm quan trọng của chuỗi quá khứ và lưu trữ chúng trong nội dung bộ nhớ cấu trúc, để khi cần có thể sử dụng dự đoán ở các bước tương lai
2. ARIMA
Mô hình tự hồi quy kết hợp trung bình trượt (AutoRegressive Integrated Moving Average)
Bắt chuỗi theo quan hệ tuyến tính giữa các quan sát trước đó
Khử xu thế (differencing) để ổn định chuỗi
Hiệu quả với chuỗi ngắn, có cấu trúc tuyến tính, ít biến động phức tạp
3. N-BEAT
Chia chuỗi thành các block hồi quy (backcast + forecast)
Mỗi block học phần dư chưa giải thích của block trước
Học trực tiếp từ dữ liệu, không cần giả định thống kê
Hiệu quả với chuỗi phi tuyến, đa dạng cấu trúc, dài-ngắn linh hoạt

⚙️ Thực nghiệm & Đánh giá
✅ Huấn luyện trên 50 epoch và phân tích chuỗi thười gian dựa trên 5 hộ có lượng tiêu thụ cao nhất
📊 Sử dụng các chỉ số: MAPE, MSE, MAE

💻 Giao diện ứng dụng
Triển khai giao diện người dùng tại:
🔗(https://dudoandiennangtieuthu.streamlit.app/)

Chức năng:

Chọn hộ cần dự báo
Tự động nạp mô hình đã huấn luyện
Hiển thị dự báo & biểu đồ mức độ tiêu thụ
📁 Cấu trúc thư mục
├── data/ # Dữ liệu đầu vào và sau xử lý ├── models/ # Định nghĩa mô hình (LSTM, ARIMA, NBEAT)├── notebook/ # Notebook huấn luyện & đánh giá mô hình ├── app.py # Giao diện streamlit  ├── README.md # File mô tả dự án

🔧 Cài đặt
git clone https://github.com/Annlt272/DuDoanDienNang.git
cd DuDoanDienNang
pip install -r requirements.txt

🧪 Hướng phát triển tương lai
Tự động hóa lựa chọn siêu tham số
Xử lý mất cân bằng nhãn
Dự báo thời gian thực và mở rộng cho nhiều vùng địa lý khác nhau
Kết hợp thêm dữ liệu dịp lễ, thời tiết, môi trường và khí hậu

👨‍💻 Nhóm thực hiện
Lương Thành An – 2251262567

Hoàng Thị Hồng – 2251262577

Lê Thị Như Quỳnh – 2251262632

Trường Đại học Thủy Lợi – Khoa Công nghệ thông tin
Lớp 64TTNT1 – Năm học 2024–2025
