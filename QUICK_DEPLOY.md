# 🚀 Quick Deploy to Vercel

## Prerequisites
- ✅ Code is ready (build tested locally)
- ✅ Vercel account (sign up at [vercel.com](https://vercel.com))
- ⚠️ **Vercel Pro plan required** (model is 300MB, free tier limit is 250MB)

## Deploy in 3 Steps

### 1️⃣ Push to Git
```bash
cd BreastCancerDetect
git init
git add .
git commit -m "Ready for deployment"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2️⃣ Import to Vercel
1. Go to [vercel.com/new](https://vercel.com/new)
2. Click "Import Git Repository"
3. Select your repository
4. **Root Directory**: `./` (leave as default)
5. Click **Deploy**

### 3️⃣ Wait & Test
- Deployment takes 3-5 minutes
- Visit your URL: `https://your-app.vercel.app`
- Test upload functionality

## That's It! 🎉

Your app is live with:
- ✅ React frontend at `/`
- ✅ FastAPI backend at `/api/*`
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Auto-deploy on git push

---

## Alternative: Vercel CLI

```bash
npm install -g vercel
cd BreastCancerDetect
vercel login
vercel --prod
```

---

## First Request Will Be Slow ⏱️

- Cold start: 10-30 seconds (loading TensorFlow + model)
- Subsequent requests: Fast (while lambda is warm)
- This is normal for serverless functions with large models

---

## Need Help?

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed guide and troubleshooting.

