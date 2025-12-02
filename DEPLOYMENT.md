# Vercel Deployment Guide

## ✅ Pre-Deployment Checklist

Your project is now **ready for Vercel deployment**! Here's what's been configured:

### ✓ Frontend (React)
- [x] Build completes successfully (`npm run build`)
- [x] `vercel-build` script added to `package.json`
- [x] Dynamic API URL detection (works locally and on Vercel)
- [x] Production build optimized and tested
- [x] Static assets in `public/` directory

### ✓ Backend (FastAPI + TensorFlow)
- [x] All dependencies in `requirements.txt`
- [x] CORS middleware configured
- [x] Model file (`breast_cancer_model.keras`) ~300MB included
- [x] Grad-CAM and PDF report generation working
- [x] Health check endpoint (`/api/health`)

### ✓ Configuration
- [x] `vercel.json` with monorepo build setup
- [x] Routes configured (`/api/*` → backend, `/*` → frontend)
- [x] Lambda settings optimized (60s timeout, 3GB memory, 350MB max size)
- [x] `.gitignore` files properly configured

---

## 🚀 Deployment Steps

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push to GitHub/GitLab/Bitbucket**
   ```bash
   cd BreastCancerDetect
   git init
   git add .
   git commit -m "Initial commit - Ready for Vercel"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Import to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import your repository
   - **Root Directory**: Leave as `./` (the `vercel.json` handles everything)
   - Click "Deploy"

3. **Configure Environment Variables** (Optional)
   - Go to Project Settings → Environment Variables
   - Add if needed:
     - `ALLOWED_ORIGINS`: Your frontend domain (or leave as `*`)
     - `MODEL_PATH`: Only if hosting model externally

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy from project root**
   ```bash
   cd BreastCancerDetect
   vercel
   ```

4. **Follow the prompts:**
   - Set up and deploy? **Y**
   - Which scope? Select your account
   - Link to existing project? **N**
   - Project name? (default or custom)
   - In which directory is your code located? **./`**
   - Deploy? **Y**

5. **For production deployment:**
   ```bash
   vercel --prod
   ```

---

## ⚠️ Important Notes

### Model Size & Vercel Plan Requirements

- Your TensorFlow model is **~300MB**
- **Vercel Pro plan or higher is REQUIRED** for this deployment
- Free tier has a 250MB serverless function limit
- Pro plan allows up to 350MB (configured in `vercel.json`)

### Cold Start Performance

- First request after inactivity may take **10-30 seconds**
- TensorFlow + model need to load into memory
- Subsequent requests will be faster (while lambda is warm)
- Consider implementing a cron job to keep the function warm

### Alternative: External Model Hosting

If you want to reduce lambda size, host the model externally:

1. Upload `breast_cancer_model.keras` to:
   - Vercel Blob Storage
   - AWS S3
   - Google Cloud Storage
   - Any CDN

2. Set `MODEL_PATH` environment variable to the download URL

3. Modify `backend/main.py` to download the model at startup:
   ```python
   import requests
   
   model_url = os.getenv("MODEL_PATH")
   if model_url.startswith("http"):
       response = requests.get(model_url)
       with open("/tmp/model.keras", "wb") as f:
           f.write(response.content)
       model = keras.models.load_model("/tmp/model.keras")
   ```

---

## 🧪 Post-Deployment Testing

Once deployed, test these endpoints:

### 1. Health Check
```bash
curl https://your-app.vercel.app/api/health
```
Expected response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. Frontend
Visit `https://your-app.vercel.app` in your browser:
- Upload page should load
- Drag & drop should work
- File browser should work

### 3. Analysis Endpoint
```bash
curl -X POST https://your-app.vercel.app/api/analyze \
  -F "file=@test_image.png"
```

### 4. PDF Report
```bash
curl -X POST https://your-app.vercel.app/api/report \
  -F "file=@test_image.png" \
  -o report.pdf
```

---

## 🐛 Troubleshooting

### Build Fails

**Frontend build error:**
```bash
cd frontend
npm install
npm run build
```
Check the error and fix any React compilation issues.

**Backend build error:**
- Verify all dependencies are in `requirements.txt`
- Check Python version compatibility (Vercel uses Python 3.9)
- Ensure TensorFlow version is compatible

### Lambda Size Exceeded

Error: `Serverless Function exceeds maximum size`

Solutions:
1. Upgrade to Vercel Pro plan
2. Host model externally (see above)
3. Use a lighter TensorFlow build (`tensorflow-cpu`)

### CORS Errors

If frontend can't reach backend:
1. Check `ALLOWED_ORIGINS` environment variable
2. Verify routes in `vercel.json` are correct
3. Check browser console for specific CORS error

### Cold Start Timeout

If requests timeout on first load:
1. Increase `maxDuration` in `vercel.json` (max 60s on Pro)
2. Implement a warming strategy (cron job hitting `/api/health`)
3. Consider splitting backend to a dedicated service (Railway, Render)

### Model Not Loading

Check logs in Vercel dashboard:
- Go to your project → Deployments → Click latest → Functions
- Look for errors in the function logs
- Verify `MODEL_PATH` is correct

---

## 📊 Monitoring

After deployment, monitor:

1. **Function Logs**: Vercel Dashboard → Functions → View logs
2. **Performance**: Check cold start times and execution duration
3. **Errors**: Set up error tracking (Sentry, LogRocket)
4. **Usage**: Monitor bandwidth and function invocations

---

## 🔄 Continuous Deployment

Once connected to Git:
- Every push to `main` triggers automatic deployment
- Preview deployments for pull requests
- Rollback to previous deployments in one click

---

## 📝 Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REACT_APP_API_BASE_URL` | No | `/api` | Frontend API base URL |
| `MODEL_PATH` | No | `backend/models/breast_cancer_model.keras` | Path to model file |
| `ALLOWED_ORIGINS` | No | `*` | CORS allowed origins |

---

## ✅ You're Ready!

Your project is fully configured and ready to deploy. Just follow the steps above and you'll have your breast cancer detection app live on Vercel in minutes!

**Need help?** Check the [Vercel documentation](https://vercel.com/docs) or open an issue in your repository.

