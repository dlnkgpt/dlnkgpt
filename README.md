# dLNk GPT - Complete Implementation
## Overview

dLNk GPT is a complete implementation of an uncensored AI chat service based on GPT-J-6B. This repository contains all the necessary code, scripts, and configurations to:

1. Fine-tune GPT-J-6B on custom datasets
2. Deploy a FastAPI backend server
3. Serve a web-based frontend interface
4. Containerize and deploy with Docker

## Project Structure

```
dlnkgpt_project/
├── model_finetuning/           # Model training scripts
│   ├── data/                   # Training data
│   │   └── training_data.jsonl # 1000 training examples
│   ├── prepare_env.py          # Download model and prepare environment
│   ├── create_dataset_only.py  # Create training dataset only
│   ├── fine_tune.py            # Main fine-tuning script
│   └── TRAINING_GUIDE.md       # Detailed training instructions
│
├── backend_api/                # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   ├── models.py           # Pydantic models
│   │   └── security.py         # Authentication & security
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Docker configuration
│   └── .dockerignore
│
├── frontend_ui/                # Web frontend
│   └── index.html              # Dark-themed chat interface
│
├── deployment/                 # Deployment configurations
│   ├── docker-compose.yml      # Docker Compose setup
│   └── nginx.conf              # Nginx reverse proxy config
│
└── .env.example                # Environment variables template
```

## Quick Start

### Prerequisites

- **Hardware:**
  - CPU: 8+ cores
  - RAM: 32 GB minimum (64 GB recommended)
  - Storage: 100 GB free space
  - GPU: NVIDIA GPU with 24+ GB VRAM (optional but recommended)

- **Software:**
  - Python 3.9+
  - Docker & Docker Compose
  - CUDA Toolkit (if using GPU)

### Step 1: Clone and Setup

```bash
cd dlnkgpt_project

# Copy environment variables
cp .env.example .env

# Edit .env with your configurations
nano .env
```

### Step 2: Install Dependencies

```bash
cd model_finetuning

# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or for GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers accelerate datasets scikit-learn
```

### Step 3: Prepare Training Data

The training dataset is already created with 1,000 examples. To regenerate:

```bash
python create_dataset_only.py
```

### Step 4: Download Base Model and Fine-tune

**⚠️ This step requires ~24 GB of disk space and may take 4-72 hours depending on hardware.**

```bash
# Download model and start fine-tuning
python fine_tune.py
```

For detailed training instructions, see [TRAINING_GUIDE.md](model_finetuning/TRAINING_GUIDE.md).

### Step 5: Run Backend API (Development)

```bash
cd ../backend_api

# Install backend dependencies
pip install -r requirements.txt

# Run the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: http://localhost:8000

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Step 6: Deploy with Docker (Production)

```bash
cd ../deployment

# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services will be available at:
- Frontend: http://localhost
- API: http://localhost/api/
- Direct API: http://localhost:8000

## API Usage

### Authentication

The API uses API key authentication. Demo keys are pre-configured:

- `demo_key_123` (Premium tier)
- `test_key_456` (Basic tier)

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "demo_key_123",
    "prompt": "Write a hello world program in Python",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### Response

```json
{
  "response": "Here is a hello world program in Python:\n\nprint('Hello, World!')",
  "tokens_used": 15
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status |
| `/health` | GET | Health check |
| `/chat` | POST | Generate response |
| `/model/info` | GET | Model information |
| `/docs` | GET | API documentation |

## Configuration

### Environment Variables

Edit `.env` file to configure:

- `MODEL_PATH`: Path to fine-tuned model
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `LOG_LEVEL`: Logging level
- `DEFAULT_MAX_TOKENS`: Default max tokens
- `DEFAULT_TEMPERATURE`: Default temperature

### Model Parameters

Edit `fine_tune.py` to adjust training parameters:

- `num_train_epochs`: Number of training epochs (default: 5)
- `per_device_train_batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 2e-5)
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors during training:

1. Reduce batch size: `per_device_train_batch_size=2`
2. Increase gradient accumulation: `gradient_accumulation_steps=8`
3. Use 8-bit quantization (requires `bitsandbytes`)

### Model Not Loading

If the model fails to load:

1. Check that `dlnkgpt-model/` directory exists
2. Verify model files are present (pytorch_model.bin, config.json)
3. Check disk space and permissions
4. Review logs for specific errors

### API Connection Issues

If you can't connect to the API:

1. Check if the service is running: `docker-compose ps`
2. Verify port mappings: `docker-compose port api 8000`
3. Check firewall settings
4. Review nginx logs: `docker-compose logs nginx`

## Performance Optimization

### GPU Acceleration

To enable GPU support in Docker:

1. Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
2. Uncomment GPU configuration in `docker-compose.yml`
3. Rebuild and restart containers

### Caching

The model is cached in `cached_model/` directory to avoid re-downloading. To clear cache:

```bash
rm -rf model_finetuning/cached_model
```


This project is provided as-is for educational purposes. Use at your own risk.

## References

- [GPT-J-6B by EleutherAI](https://huggingface.co/EleutherAI/gpt-j-6b)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

## Support

For issues and questions:

1. Check the [TRAINING_GUIDE.md](model_finetuning/TRAINING_GUIDE.md)
2. Review troubleshooting section above
3. Check logs for error messages
4. Consult the API documentation at `/docs`

---

**Version:** 1.0.0  
**Last Updated:** 2025-01-13  
**Status:** Complete Implementation
