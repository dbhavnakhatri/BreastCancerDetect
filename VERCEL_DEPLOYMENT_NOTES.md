# Vercel Deployment - Important Notes

## ✅ Changes Made to Fix Deployment

### 1. **Python Version**
- Changed from Python 3.12 to **Python 3.11**
- Python 3.12 removed `distutils` which breaks many packages
- Added `api/runtime.txt` to specify Python 3.11

### 2. **Package Versions**
- Updated to Python 3.11-compatible versions
- Changed `tensorflow` to `tensorflow-cpu` (smaller, no GPU support needed on serverless)
- Updated numpy, scipy, matplotlib to compatible versions

### 3. **Current Status**
✅ Frontend builds successfully  
⚠️ Backend may still face size limits

---

## ⚠️ Critical Issue: TensorFlow Model Size

Your model (`breast_cancer_model.keras`) is **~300MB**, and with TensorFlow dependencies, the total function size will exceed **500MB**.

### Vercel Limits:
- **Free Plan**: 250MB max
- **Pro Plan**: 350MB max (still too small!)
- **Enterprise Plan**: Custom limits

---

## 🚀 Solutions (Choose One)

### **Option 1: Deploy Backend Separately (RECOMMENDED)**

Deploy your backend to a platform that supports larger applications:

#### **A. Railway.app** (Easiest)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Deploy backend
cd backend
railway init
railway up
```

Then update your Vercel frontend to point to Railway:
- Set `REACT_APP_API_BASE_URL` in Vercel to your Railway URL

#### **B. Render.com** (Free tier available)
1. Go to [render.com](https://render.com)
2. Create new "Web Service"
3. Connect your GitHub repo
4. Root Directory: `backend`
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### **C. Google Cloud Run / AWS Lambda**
For production-grade deployment with auto-scaling.

---

### **Option 2: Host Model Externally**

Keep backend on Vercel but host the model file separately:

#### **Step 1: Upload Model to Cloud Storage**

**Google Cloud Storage:**
```bash
# Install gcloud CLI
gsutil cp backend/models/breast_cancer_model.keras gs://your-bucket/
gsutil acl ch -u AllUsers:R gs://your-bucket/breast_cancer_model.keras
```

**AWS S3:**
```bash
aws s3 cp backend/models/breast_cancer_model.keras s3://your-bucket/ --acl public-read
```

**Vercel Blob Storage:**
```bash
npm i -g vercel
vercel blob upload backend/models/breast_cancer_model.keras
```

#### **Step 2: Modify Backend to Download Model**

Update `backend/main.py`:

```python
import os
import requests
from pathlib import Path

MODEL_URL = os.getenv("MODEL_URL", "https://your-storage-url/breast_cancer_model.keras")
MODEL_PATH = "/tmp/breast_cancer_model.keras"

def load_model_from_url():
    if not Path(MODEL_PATH).exists():
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully")
    return keras.models.load_model(MODEL_PATH)

# Replace the existing model loading
model = load_model_from_url()
```

#### **Step 3: Set Environment Variable in Vercel**
- Go to Project Settings → Environment Variables
- Add: `MODEL_URL` = `https://your-storage-url/breast_cancer_model.keras`

---

### **Option 3: Use Lighter Model (If Possible)**

If you can retrain or compress your model:

1. **Model Quantization** (reduces size by 4x):
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

2. **Model Pruning** (removes unnecessary weights)
3. **Knowledge Distillation** (train smaller model from larger one)

---

## 📊 Recommended Architecture

### **Best Practice: Split Deployment**

```
┌─────────────────┐
│  Vercel         │
│  (Frontend)     │ ──────┐
└─────────────────┘       │
                          │ HTTPS
┌─────────────────┐       │
│  Railway/Render │ ◄─────┘
│  (Backend+Model)│
└─────────────────┘
```

**Benefits:**
- ✅ No size limits
- ✅ Better performance (dedicated resources)
- ✅ Easier debugging
- ✅ Can scale independently
- ✅ Railway/Render have free tiers

---

## 🔧 Quick Fix: Deploy to Railway

**5-Minute Setup:**

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Create new project
cd backend
railway init

# 4. Deploy
railway up

# 5. Get your URL
railway domain

# 6. Update Vercel environment variable
# REACT_APP_API_BASE_URL = https://your-app.railway.app
```

Then redeploy your Vercel frontend - it will now connect to Railway backend!

---

## 📝 Current Deployment Status

- ✅ Frontend: Ready for Vercel
- ⚠️ Backend: Too large for Vercel (needs Railway/Render or external model hosting)

---

## 🆘 Need Help?

1. **Railway Tutorial**: https://docs.railway.app/getting-started
2. **Render Tutorial**: https://render.com/docs/deploy-fastapi
3. **Vercel Blob Storage**: https://vercel.com/docs/storage/vercel-blob

Choose the solution that best fits your needs. I recommend **Railway** for the easiest setup!

