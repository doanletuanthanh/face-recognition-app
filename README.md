# Face Recognition App

Thực hiện lại Homework môn học CS331-Thị giác máy tính nâng cao xây dựng app nhận diện khuôn mặt sử dụng Flask (frontend) và FastAPI (backend), hỗ trợ nhận diện từ ảnh, webcam, và thêm khuôn mặt mới vào database.

## Cấu trúc thư mục

```
face_index.pkl
face.index
requirements.txt
run_flask.py
app/
    __init__.py
    detection.py
    face_db.py
    main.py
    playground.py
    recognition.py
flask_app/
    __init__.py
    routes.py
    templates/
model2onnx/
    convert_onnx.py
    ...
models/
    arcface_r100_v1.onnx
    ...
```

## Công nghệ sử dụng

- **YOLO**: Phát hiện khuôn mặt nhanh và chính xác trên ảnh và video.
- **ArcFace**: Trích xuất embedding khuôn mặt với độ chính xác cao, phục vụ nhận diện.
- **Flask & FastAPI**: Kết hợp xây dựng giao diện demo đơn giản.

## Cài đặt

1. **Clone repo và cài đặt thư viện:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Chuẩn bị model:**
   - Đảm bảo các file `.onnx` và model nhận diện khuôn mặt đã có trong thư mục `models/`.

## Chạy ứng dụng

### 1. Chạy backend (FastAPI)

```sh
uvicorn app.main:app --reload
```

### 2. Chạy frontend (Flask)

```sh
python run_flask.py
```

- Flask mặc định chạy ở `http://127.0.0.1:5000`
- FastAPI backend chạy ở `http://127.0.0.1:8000`

## Demo sử dụng

1. **Truy cập giao diện:**  
   Mở trình duyệt và vào [http://localhost:5000](http://localhost:5000)

2. **Các chức năng:**
   - **Add Face:**  
     Chụp ảnh từ webcam, nhập tên và thêm vào database.
   - **Image Recognize:**  
     Upload ảnh để nhận diện khuôn mặt.
   - **Live Recognize:**  
     Nhận diện khuôn mặt trực tiếp từ webcam.

3. **Lưu ý:**
   - Khi thêm khuôn mặt mới, đảm bảo chỉ có 1 khuôn mặt trong khung hình.

