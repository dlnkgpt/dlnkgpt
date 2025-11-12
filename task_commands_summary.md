# สรุปคำสั่งสำหรับแต่ละ Task - dLNk GPT

เอกสารนี้เป็นเวอร์ชันย่อที่รวบรวมคำสั่งที่จำเป็นสำหรับแต่ละ Task ไว้ในที่เดียว คุณสามารถคัดลอกและวางคำสั่งเหล่านี้ไปยัง Manus ได้ทันที

---

## Task 1: เตรียมโครงสร้างโปรเจค

```
ฉันต้องการเริ่มต้นโปรเจค dLNk GPT โปรดดำเนินการดังนี้:

1. สร้างโครงสร้างไดเรกทอรีทั้งหมดตามนี้:
   - /home/ubuntu/dlnkgpt_project/
   - /home/ubuntu/dlnkgpt_project/model_finetuning/data/
   - /home/ubuntu/dlnkgpt_project/backend_api/app/
   - /home/ubuntu/dlnkgpt_project/frontend_ui/
   - /home/ubuntu/dlnkgpt_project/deployment/

2. สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/backend_api/requirements.txt` และใส่เนื้อหาดังต่อไปนี้:

fastapi
uvicorn[standard]
pydantic
sqlalchemy
psycopg2-binary
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
transformers
torch
accelerate
datasets
scikit-learn
```

---

## Task 2: ดาวน์โหลดโมเดลและสร้างชุดข้อมูล

```
โปรดสร้างและรันสคริปต์ Python เพื่อเตรียมสภาพแวดล้อมสำหรับการฝึกโมเดล dLNk GPT โดยทำตามขั้นตอนต่อไปนี้:

1. สร้างไฟล์ Python ชื่อ `/home/ubuntu/dlnkgpt_project/model_finetuning/prepare_env.py` ที่มีฟังก์ชัน:
   - download_model(): ดาวน์โหลด EleutherAI/gpt-j-6b และ tokenizer
   - create_simulated_dataset(): สร้างไฟล์ training_data.jsonl ที่มี 1000 ตัวอย่าง

2. รันสคริปต์โดยใช้คำสั่ง: `python3.11 /home/ubuntu/dlnkgpt_project/model_finetuning/prepare_env.py`
```

---

## Task 3: สร้างสคริปต์ Fine-Tuning

```
โปรดสร้างสคริปต์สำหรับ Fine-tuning โมเดล dLNk GPT:

สร้างไฟล์ชื่อ `/home/ubuntu/dlnkgpt_project/model_finetuning/fine_tune.py` ที่มีฟังก์ชันหลักดังนี้:
- โหลด tokenizer และ model จาก EleutherAI/gpt-j-6b
- โหลดและ tokenize dataset จาก training_data.jsonl
- กำหนด TrainingArguments โดยใช้:
  * num_train_epochs=5
  * per_device_train_batch_size=4
  * learning_rate=2e-5
- สร้าง Trainer และเริ่มการฝึก
- บันทึกโมเดลที่ฝึกเสร็จไปที่ /home/ubuntu/dlnkgpt_project/model_finetuning/dlnkgpt-model
```

---

## Task 4: รัน Fine-Tuning

```
โปรดเริ่มกระบวนการ Fine-tuning โมเดล dLNk GPT โดยรันสคริปต์ที่เตรียมไว้ใน Task ที่แล้ว

ใช้คำสั่ง: `python3.11 /home/ubuntu/dlnkgpt_project/model_finetuning/fine_tune.py`

ฉันเข้าใจว่าขั้นตอนนี้อาจใช้เวลานานในการดำเนินการ
```

---

## Task 5: สร้าง Backend API

```
โปรดสร้างไฟล์สำหรับ Backend API ของโปรเจค dLNk GPT ดังนี้:

1. สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/backend_api/app/models.py` ที่มี:
   - ChatRequest (api_key, prompt)
   - ChatResponse (response)

2. สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/backend_api/app/security.py` ที่มี:
   - validate_api_key()
   - check_subscription()

3. สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/backend_api/app/main.py` ที่มี:
   - FastAPI app
   - ModelSingleton class สำหรับโหลดโมเดล
   - POST /chat endpoint ที่ใช้โมเดลในการสร้างคำตอบ
   - GET / endpoint สำหรับตรวจสอบสถานะ
```

---

## Task 6: สร้าง Frontend Placeholder

```
โปรดสร้างไฟล์ Placeholder สำหรับ Frontend ของโปรเจค dLNk GPT:

สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/frontend_ui/index.html` ที่แสดงข้อความ "dLNk GPT - Frontend UI is under construction" พร้อมกับ dark theme
```

---

## Task 7: สร้างไฟล์ Deployment

```
โปรดสร้างไฟล์สำหรับการ Deployment โปรเจค dLNk GPT ด้วย Docker:

1. สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/backend_api/Dockerfile` ที่:
   - ใช้ base image python:3.9-slim
   - คัดลอก requirements.txt และติดตั้ง dependencies
   - คัดลอก app และโมเดลที่ฝึกเสร็จแล้ว
   - Expose port 8000
   - รัน uvicorn

2. สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/deployment/docker-compose.yml` ที่มี:
   - api service (build จาก backend_api)
   - nginx service (reverse proxy)

3. สร้างไฟล์ `/home/ubuntu/dlnkgpt_project/deployment/nginx.conf` ที่:
   - Proxy /api/ ไปยัง api:8000
   - Serve frontend files จาก /
```

---

## การรันระบบหลังจากเสร็จสิ้นทุก Task

เมื่อดำเนินการทุก Task เสร็จสิ้นแล้ว คุณสามารถรันระบบด้วยคำสั่ง:

```bash
cd /home/ubuntu/dlnkgpt_project/deployment
docker-compose up -d
```

จากนั้นเข้าถึง API ผ่าน: `http://localhost/api/`
