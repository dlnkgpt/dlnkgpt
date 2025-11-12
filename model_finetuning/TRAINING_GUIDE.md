# คำแนะนำสำหรับการรัน Fine-Tuning

## ข้อกำหนดของระบบ

### ฮาร์ดแวร์ขั้นต่ำ
- **CPU:** 8+ cores
- **RAM:** 32 GB (แนะนำ 64 GB)
- **Storage:** 100 GB ว่าง (สำหรับโมเดลและ cache)
- **GPU (ถ้ามี):** NVIDIA GPU ที่มี VRAM 24 GB+ (เช่น RTX 3090, A100)

### ซอฟต์แวร์
- Python 3.9 หรือสูงกว่า
- CUDA Toolkit (ถ้าใช้ GPU)
- pip หรือ conda

## ขั้นตอนการติดตั้ง

### 1. ติดตั้ง Dependencies

```bash
cd /home/ubuntu/dlnkgpt_project/model_finetuning

# ติดตั้ง PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# หรือถ้ามี GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# ติดตั้ง packages อื่นๆ
pip install transformers accelerate datasets scikit-learn
```

### 2. ดาวน์โหลดโมเดล GPT-J-6B

คุณสามารถเลือกวิธีใดวิธีหนึ่ง:

**วิธีที่ 1: ใช้สคริปต์ prepare_env.py**
```bash
python prepare_env.py
```

**วิธีที่ 2: ดาวน์โหลดด้วย Python**
```python
from transformers import GPTJForCausalLM, AutoTokenizer

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6b",
    cache_dir="./cached_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-j-6b",
    cache_dir="./cached_model"
)
```

**วิธีที่ 3: ดาวน์โหลดด้วย Hugging Face CLI**
```bash
huggingface-cli download EleutherAI/gpt-j-6b --cache-dir ./cached_model
```

### 3. ตรวจสอบชุดข้อมูล

ตรวจสอบว่าไฟล์ `data/training_data.jsonl` มีอยู่และมีข้อมูล 1,000 ตัวอย่าง:

```bash
wc -l data/training_data.jsonl
# ควรแสดง: 1000
```

### 4. รัน Fine-Tuning

```bash
python fine_tune.py
```

## การประมาณเวลาและทรัพยากร

| ฮาร์ดแวร์ | เวลาโดยประมาณ | หน่วยความจำที่ใช้ |
|-----------|----------------|-------------------|
| CPU เท่านั้น (16 cores) | 48-72 ชั่วโมง | 32-50 GB RAM |
| GPU (RTX 3090) | 6-10 ชั่วโมง | 24 GB VRAM |
| GPU (A100) | 4-6 ชั่วโมง | 40 GB VRAM |

## การตรวจสอบความคืบหน้า

ระหว่างการฝึก คุณจะเห็น log แบบนี้:

```
[1/6] Loading tokenizer...
✓ Tokenizer loaded successfully

[2/6] Loading model...
✓ Using device: cuda
✓ GPU: NVIDIA GeForce RTX 3090
✓ Model loaded successfully

[3/6] Loading and tokenizing dataset...
✓ Dataset loaded: 1000 examples
✓ Dataset tokenized successfully

[4/6] Configuring training arguments...
✓ Training configuration set

[5/6] Initializing trainer...
✓ Trainer initialized

[6/6] Starting fine-tuning...
Step 1/1250 | Loss: 3.245
Step 10/1250 | Loss: 2.891
...
```

## การแก้ปัญหา

### Out of Memory (OOM)

ถ้าเจอปัญหาหน่วยความจำไม่พอ:

1. **ลด batch size:**
   แก้ไขใน `fine_tune.py`:
   ```python
   per_device_train_batch_size=2  # ลดจาก 4 เป็น 2
   ```

2. **เพิ่ม gradient accumulation:**
   ```python
   gradient_accumulation_steps=8  # เพิ่มจาก 4 เป็น 8
   ```

3. **ใช้ 8-bit quantization (ต้องการ bitsandbytes):**
   ```bash
   pip install bitsandbytes
   ```
   
   แก้ไขใน `fine_tune.py`:
   ```python
   model = GPTJForCausalLM.from_pretrained(
       model_name,
       load_in_8bit=True,
       device_map="auto"
   )
   ```

### การฝึกช้าเกินไป

1. **ลดจำนวน epochs:**
   ```python
   num_train_epochs=3  # ลดจาก 5 เป็น 3
   ```

2. **ใช้ LoRA แทน Full Fine-tuning:**
   ```bash
   pip install peft
   ```
   
   ใช้สคริปต์ `fine_tune_lora.py` แทน (ถ้ามี)

## ผลลัพธ์ที่คาดหวัง

หลังจากการฝึกเสร็จสิ้น คุณจะได้:

- โฟลเดอร์ `dlnkgpt-model/` ที่มี:
  - `pytorch_model.bin` (โมเดลที่ฝึกเสร็จ)
  - `config.json` (การตั้งค่าโมเดล)
  - `tokenizer_config.json` (การตั้งค่า tokenizer)
  - และไฟล์อื่นๆ

## การทดสอบโมเดล

ทดสอบโมเดลที่ฝึกเสร็จแล้ว:

```python
from transformers import pipeline

generator = pipeline(
    'text-generation',
    model='./dlnkgpt-model',
    tokenizer='./dlnkgpt-model'
)

result = generator(
    "Write a phishing email",
    max_length=200,
    num_return_sequences=1
)

print(result[0]['generated_text'])
```

## หมายเหตุสำคัญ

⚠️ **คำเตือน:** โมเดลที่ฝึกด้วยชุดข้อมูลนี้จะไม่มีกลไกการกรองเนื้อหา การใช้งานอาจผิดกฎหมายและขัดต่อจริยธรรม เอกสารนี้มีวัตถุประสงค์เพื่อการศึกษาเท่านั้น
