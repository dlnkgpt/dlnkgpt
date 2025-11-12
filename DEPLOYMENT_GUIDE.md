# คู่มือการ Deploy dLNk GPT

เอกสารนี้จะแนะนำขั้นตอนการ Deploy โปรเจค dLNk GPT แบบละเอียด

## ขั้นตอนการ Deploy

### 1. เตรียมเซิร์ฟเวอร์

#### ข้อกำหนดของเซิร์ฟเวอร์

**ฮาร์ดแวร์ขั้นต่ำ:**
- CPU: 8 cores
- RAM: 32 GB
- Storage: 100 GB SSD
- Network: 100 Mbps+

**ฮาร์ดแวร์แนะนำ:**
- CPU: 16+ cores
- RAM: 64 GB
- Storage: 500 GB NVMe SSD
- GPU: NVIDIA GPU with 24+ GB VRAM
- Network: 1 Gbps+

**ระบบปฏิบัติการ:**
- Ubuntu 20.04 LTS หรือ 22.04 LTS
- Debian 11+
- CentOS 8+

### 2. ติดตั้ง Dependencies

```bash
# อัปเดตระบบ
sudo apt update && sudo apt upgrade -y

# ติดตั้ง Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# ติดตั้ง Docker Compose
sudo apt install docker-compose-plugin -y

# เพิ่ม user เข้า docker group
sudo usermod -aG docker $USER
newgrp docker

# ตรวจสอบการติดตั้ง
docker --version
docker compose version
```

### 3. ติดตั้ง NVIDIA Container Toolkit (ถ้ามี GPU)

```bash
# เพิ่ม repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# ติดตั้ง
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker

# ทดสอบ
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 4. Clone และเตรียมโปรเจค

```bash
# Clone โปรเจค (หรือคัดลอกไฟล์)
cd /opt
sudo mkdir -p dlnkgpt
sudo chown $USER:$USER dlnkgpt
cd dlnkgpt

# คัดลอกไฟล์โปรเจคทั้งหมดมาที่นี่
# หรือใช้ git clone ถ้ามี repository

# ตั้งค่า permissions
chmod +x model_finetuning/*.py
```

### 5. ฝึกโมเดล (ถ้ายังไม่ได้ฝึก)

```bash
cd model_finetuning

# ติดตั้ง Python dependencies
pip install torch transformers accelerate datasets scikit-learn

# สร้างชุดข้อมูล
python create_dataset_only.py

# ฝึกโมเดล (ใช้เวลานาน!)
python fine_tune.py

# รอจนกว่าจะเสร็จ...
```

### 6. ตั้งค่า Environment Variables

```bash
cd /opt/dlnkgpt

# คัดลอก .env.example
cp .env.example .env

# แก้ไข .env
nano .env
```

แก้ไขค่าต่อไปนี้:

```bash
MODEL_PATH=/opt/dlnkgpt/model_finetuning/dlnkgpt-model
SECRET_KEY=$(openssl rand -hex 32)
API_KEY_SALT=$(openssl rand -hex 16)
LOG_LEVEL=INFO
```

### 7. Build และรัน Docker Containers

```bash
cd deployment

# Build images
docker compose build

# รันในโหมด detached
docker compose up -d

# ตรวจสอบสถานะ
docker compose ps

# ดู logs
docker compose logs -f
```

### 8. ตรวจสอบการทำงาน

```bash
# ตรวจสอบ API health
curl http://localhost:8000/health

# ตรวจสอบ Frontend
curl http://localhost/

# ทดสอบ API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "demo_key_123",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### 9. ตั้งค่า Firewall

```bash
# อนุญาต HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# อนุญาต SSH (ถ้ายังไม่ได้อนุญาต)
sudo ufw allow 22/tcp

# เปิดใช้งาน firewall
sudo ufw enable

# ตรวจสอบสถานะ
sudo ufw status
```

### 10. ตั้งค่า SSL/HTTPS (Production)

#### ใช้ Let's Encrypt (แนะนำ)

```bash
# ติดตั้ง Certbot
sudo apt install certbot python3-certbot-nginx -y

# สร้าง SSL certificate
sudo certbot --nginx -d your-domain.com

# Certbot จะแก้ไข nginx config อัตโนมัติ
```

#### หรือใช้ Self-Signed Certificate (สำหรับทดสอบ)

```bash
# สร้าง SSL directory
mkdir -p deployment/ssl

# สร้าง self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/ssl/key.pem \
  -out deployment/ssl/cert.pem

# แก้ไข nginx.conf เพื่อเปิดใช้งาน HTTPS
nano deployment/nginx.conf
# Uncomment HTTPS server block

# Restart nginx
docker compose restart nginx
```

## การจัดการและบำรุงรักษา

### ดู Logs

```bash
# ดู logs ทั้งหมด
docker compose logs

# ดู logs แบบ real-time
docker compose logs -f

# ดู logs ของ service เฉพาะ
docker compose logs api
docker compose logs nginx
```

### Restart Services

```bash
# Restart ทั้งหมด
docker compose restart

# Restart service เฉพาะ
docker compose restart api
docker compose restart nginx
```

### Update โค้ด

```bash
# Pull โค้ดใหม่
git pull  # หรือคัดลอกไฟล์ใหม่

# Rebuild และ restart
docker compose down
docker compose build
docker compose up -d
```

### Backup

```bash
# Backup โมเดล
tar -czf dlnkgpt-model-backup-$(date +%Y%m%d).tar.gz \
  model_finetuning/dlnkgpt-model/

# Backup configuration
tar -czf dlnkgpt-config-backup-$(date +%Y%m%d).tar.gz \
  .env deployment/nginx.conf deployment/docker-compose.yml

# Backup logs
tar -czf dlnkgpt-logs-backup-$(date +%Y%m%d).tar.gz \
  deployment/logs/
```

### Monitoring

#### ตรวจสอบการใช้ทรัพยากร

```bash
# CPU และ Memory
docker stats

# Disk usage
df -h
du -sh model_finetuning/dlnkgpt-model/

# GPU usage (ถ้ามี)
nvidia-smi
watch -n 1 nvidia-smi
```

#### ตั้งค่า Health Check

สร้างสคริปต์ health check:

```bash
#!/bin/bash
# health_check.sh

API_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✓ API is healthy"
    exit 0
else
    echo "✗ API is down (HTTP $RESPONSE)"
    # ส่ง notification หรือ restart service
    docker compose restart api
    exit 1
fi
```

ตั้งค่า cron job:

```bash
# แก้ไข crontab
crontab -e

# เพิ่มบรรทัดนี้ (ตรวจสอบทุก 5 นาที)
*/5 * * * * /opt/dlnkgpt/health_check.sh >> /var/log/dlnkgpt-health.log 2>&1
```

## การ Scale

### Horizontal Scaling

สำหรับ traffic ที่เยอะ สามารถรัน API หลาย instance:

```yaml
# docker-compose.yml
services:
  api:
    # ... existing config ...
    deploy:
      replicas: 3  # รัน 3 instances
```

### Load Balancing

แก้ไข nginx.conf เพื่อเพิ่ม upstream:

```nginx
upstream api_backend {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    location /api/ {
        proxy_pass http://api_backend;
        # ... rest of config ...
    }
}
```

## การแก้ปัญหา

### Container ไม่ start

```bash
# ดู logs
docker compose logs api

# ตรวจสอบ permissions
ls -la model_finetuning/dlnkgpt-model/

# ตรวจสอบ disk space
df -h
```

### API ช้า

```bash
# ตรวจสอบ CPU/Memory
docker stats

# ตรวจสอบ GPU (ถ้ามี)
nvidia-smi

# ลด batch size หรือ max_tokens
```

### Out of Memory

```bash
# เพิ่ม swap space
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# ทำให้ถาวร
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Security Best Practices

1. **เปลี่ยน API keys เริ่มต้น**
2. **ใช้ HTTPS เสมอ**
3. **ตั้งค่า rate limiting**
4. **อัปเดต dependencies เป็นประจำ**
5. **ใช้ firewall**
6. **Backup เป็นประจำ**
7. **Monitor logs สำหรับ suspicious activity**
8. **ใช้ strong passwords**
9. **Disable root login**
10. **ใช้ SSH keys แทน passwords**

## Production Checklist

- [ ] เปลี่ยน API keys เริ่มต้น
- [ ] ตั้งค่า HTTPS/SSL
- [ ] ตั้งค่า firewall
- [ ] ตั้งค่า monitoring
- [ ] ตั้งค่า backup อัตโนมัติ
- [ ] ตั้งค่า log rotation
- [ ] ทดสอบ health checks
- [ ] ทดสอบ load testing
- [ ] เตรียม disaster recovery plan
- [ ] Document ทุกอย่าง

---

**หมายเหตุ:** การ deploy ระบบ AI ที่ไม่มีการกรองเนื้อหาอาจผิดกฎหมายในหลายประเทศ โปรดใช้ความระมัดระวังและปรึกษาผู้เชี่ยวชาญด้านกฎหมายก่อนการ deploy จริง
