# Quick Start Guide - dLNk GPT

เริ่มต้นใช้งาน dLNk GPT ภายใน 10 นาที (ไม่รวมเวลาฝึกโมเดล)

## สำหรับผู้ที่มีโมเดลฝึกเสร็จแล้ว

### 1. ติดตั้ง Dependencies

```bash
cd backend_api
pip install -r requirements.txt
```

### 2. รัน API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. ทดสอบ API

เปิด browser ไปที่:
- API Docs: http://localhost:8000/docs
- Frontend: เปิดไฟล์ `frontend_ui/index.html` ใน browser

หรือใช้ curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "demo_key_123",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

## สำหรับผู้ที่ยังไม่มีโมเดล

### 1. สร้างชุดข้อมูล

```bash
cd model_finetuning
python create_dataset_only.py
```

### 2. ติดตั้ง ML Libraries

```bash
pip install torch transformers accelerate datasets scikit-learn
```

### 3. ฝึกโมเดล

```bash
python fine_tune.py
```

⚠️ **หมายเหตุ:** การฝึกโมเดลใช้เวลา 4-72 ชั่วโมง ขึ้นอยู่กับฮาร์ดแวร์

### 4. รัน API (หลังฝึกเสร็จ)

```bash
cd ../backend_api
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deploy ด้วย Docker

### 1. Build และรัน

```bash
cd deployment
docker compose up -d
```

### 2. ตรวจสอบสถานะ

```bash
docker compose ps
docker compose logs -f
```

### 3. เข้าใช้งาน

- Frontend: http://localhost
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## ทดสอบด้วย Test Script

```bash
cd /home/ubuntu/dlnkgpt_project
python test_api.py
```

## API Keys สำหรับทดสอบ

- `demo_key_123` - Premium tier
- `test_key_456` - Basic tier

## ปัญหาที่พบบ่อย

### API ไม่ start

```bash
# ตรวจสอบ logs
docker compose logs api

# หรือถ้ารันโดยตรง
# ตรวจสอบว่าติดตั้ง dependencies ครบหรือไม่
```

### โมเดลไม่โหลด

API จะทำงานใน "placeholder mode" ถ้าไม่พบโมเดล คุณยังสามารถทดสอบ API ได้แต่จะได้ response จำลอง

### Port ถูกใช้แล้ว

```bash
# เปลี่ยน port ใน docker-compose.yml
ports:
  - "8080:8000"  # เปลี่ยนจาก 8000 เป็น 8080
```

## เอกสารเพิ่มเติม

- [README.md](README.md) - เอกสารหลักฉบับสมบูรณ์
- [TRAINING_GUIDE.md](model_finetuning/TRAINING_GUIDE.md) - คู่มือการฝึกโมเดล
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - คู่มือการ deploy แบบละเอียด

## คำเตือน

⚠️ โปรเจคนี้เป็นเพื่อการศึกษาเท่านั้น การใช้งานจริงอาจผิดกฎหมาย
