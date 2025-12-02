# 🚂 Deploy Backend to Railway (RECOMMENDED)

## Why Railway?

❌ **Vercel Cannot Handle Your Backend** because:
- TensorFlow + Model = ~500MB
- Vercel Pro limit = 350MB
- Python 3.12 breaks many packages
- Serverless has cold start issues with large models

✅ **Railway Can Handle It** because:
- No size limits
- Persistent containers (no cold starts)
- Free tier available ($5 credit/month)
- Easy deployment
- Better for ML workloads

---

## 🚀 Deploy in 5 Minutes

### **Method 1: Railway Dashboard (Easiest)**

#### Step 1: Sign Up
1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Sign in with GitHub

#### Step 2: Deploy from GitHub
1. Click "Deploy from GitHub repo"
2. Select your repository: `dbhavnakhatri/BreastCancerDetect`
3. Click "Deploy Now"

#### Step 3: Configure
1. After deployment starts, click on your service
2. Go to "Settings" tab
3. **Root Directory**: Set to `backend`
4. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Click "Generate Domain" to get your public URL

#### Step 4: Update Frontend
1. Copy your Railway URL (e.g., `https://breastcancerdetect-production.up.railway.app`)
2. Go to Vercel dashboard → Your project → Settings → Environment Variables
3. Add: `REACT_APP_API_BASE_URL` = `https://your-railway-url.railway.app`
4. Redeploy Vercel

---

### **Method 2: Railway CLI (For Developers)**

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Navigate to backend
cd backend

# 4. Initialize Railway project
railway init

# 5. Link to your Railway account
railway link

# 6. Deploy
railway up

# 7. Generate public domain
railway domain

# 8. Open your app
railway open
```

---

## 📋 Post-Deployment Checklist

### ✅ Test Your Railway Backend

```bash
# 1. Health check
curl https://your-app.railway.app/health

# Expected response:
# {"status":"ok","model_loaded":true}

# 2. Test analyze endpoint
curl -X POST https://your-app.railway.app/analyze \
  -F "file=@test_image.png"

# 3. Open API docs
# Visit: https://your-app.railway.app/docs
```

### ✅ Update Vercel Frontend

1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Select your project: `breastcancerbhavnaa`
3. Settings → Environment Variables
4. Add new variable:
   - **Name**: `REACT_APP_API_BASE_URL`
   - **Value**: `https://your-railway-url.railway.app`
   - **Environment**: Production, Preview, Development (select all)
5. Click "Save"
6. Go to Deployments → Click "..." → Redeploy

---

## 🎯 Final Architecture

```
┌──────────────────────┐
│   User's Browser     │
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│  Vercel (Frontend)   │  ← React app, static files
│  breastcancer...     │
│  .vercel.app         │
└──────────┬───────────┘
           │ API calls to
           │ REACT_APP_API_BASE_URL
           ↓
┌──────────────────────┐
│  Railway (Backend)   │  ← FastAPI + TensorFlow + Model
│  breastcancer...     │
│  .railway.app        │
└──────────────────────┘
```

---

## 💰 Railway Pricing

- **Free Tier**: $5 credit/month (enough for hobby projects)
- **Developer Plan**: $5/month for $5 credit
- **Team Plan**: $20/month for $20 credit

Your app will cost approximately:
- **Idle**: $0/month (Railway sleeps inactive apps)
- **Active**: ~$3-5/month depending on usage

---

## 🔧 Troubleshooting

### Railway Build Fails

**Check build logs:**
1. Railway dashboard → Your service → Deployments
2. Click on the failed deployment
3. Check "Build Logs" tab

**Common fixes:**
```bash
# If requirements.txt not found
# Make sure Root Directory is set to "backend"

# If port binding fails
# Railway automatically sets $PORT, our Procfile handles this
```

### Frontend Can't Reach Backend

**Check CORS:**
```python
# In backend/main.py, verify:
allow_origins=["*"]  # Or add your Vercel domain
```

**Check environment variable:**
```bash
# In Vercel, verify:
REACT_APP_API_BASE_URL=https://your-railway-url.railway.app
# (no trailing slash)
```

### Model Not Loading

**Check Railway logs:**
```bash
railway logs
```

**Verify model file exists:**
```bash
# Should be in: backend/models/breast_cancer_model.keras
```

---

## 🎉 Success!

Once deployed, you'll have:
- ✅ Frontend on Vercel (fast, global CDN)
- ✅ Backend on Railway (no size limits, persistent)
- ✅ Full ML functionality working
- ✅ Automatic HTTPS on both
- ✅ Auto-deploy on git push

---

## 📚 Additional Resources

- [Railway Documentation](https://docs.railway.app/)
- [Railway Discord](https://discord.gg/railway) - Very helpful community
- [Railway Templates](https://railway.app/templates) - Pre-configured setups

---

## 🆘 Still Having Issues?

1. Check Railway logs: `railway logs`
2. Check Vercel logs: Vercel Dashboard → Functions
3. Verify environment variables are set correctly
4. Make sure both deployments are using latest code

**Common mistake:** Forgetting to redeploy Vercel after adding environment variable!

---

## ⚡ Quick Commands Reference

```bash
# Railway
railway login                 # Login to Railway
railway link                  # Link to existing project
railway up                    # Deploy
railway logs                  # View logs
railway open                  # Open in browser
railway domain                # Manage domains

# Update after code changes
git add .
git commit -m "Update"
git push                      # Railway auto-deploys!
```

---

**You're almost there! Just deploy to Railway and update the Vercel environment variable.** 🚀

