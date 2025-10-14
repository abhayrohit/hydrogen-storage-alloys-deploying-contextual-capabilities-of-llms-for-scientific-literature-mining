# 🚀 GitHub Pages Deployment Guide

## Step 1: Enable GitHub Pages

1. Go to your GitHub repository: https://github.com/arooon-n/literature-mining-hydrogen-storage-alloys
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Source**, select:
   - **Source**: GitHub Actions
4. Click **Save**

## Step 2: Push Your Code

Run these commands in PowerShell from your project directory:

```powershell
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Deploy frontend to GitHub Pages"

# Add remote (if not already added)
git remote add origin https://github.com/arooon-n/literature-mining-hydrogen-storage-alloys.git

# Push to main branch
git push -u origin main
```

## Step 3: Wait for Deployment

1. Go to **Actions** tab in your GitHub repository
2. You'll see "Deploy Frontend to GitHub Pages" workflow running
3. Wait 1-2 minutes for it to complete
4. Once done, your frontend will be live at:
   **https://arooon-n.github.io/literature-mining-hydrogen-storage-alloys/**

## Step 4: Use the Application

### Start Backend Locally:
```powershell
python main_fastapi.py
```

### Access Frontend:
Open in browser: https://arooon-n.github.io/literature-mining-hydrogen-storage-alloys/

The GitHub Pages frontend will automatically connect to your local backend at http://localhost:8000

## 🔄 Updating the Deployment

Any time you push changes to the `main` branch, GitHub Actions will automatically redeploy:

```powershell
git add .
git commit -m "Update frontend"
git push origin main
```

## ⚠️ Important Notes

1. **Backend Must Run Locally**: GitHub Pages only hosts the frontend (HTML/CSS/JS). You must run the Python backend on your machine.

2. **CORS is Configured**: The backend already allows requests from any origin, so GitHub Pages can connect to your local backend.

3. **First Time Setup**: 
   - If you get authentication errors, set up GitHub credentials:
     ```powershell
     git config --global user.name "Your Name"
     git config --global user.email "your-email@example.com"
     ```
   - You may need to generate a Personal Access Token (PAT) from GitHub Settings → Developer settings → Personal access tokens

## 🎯 Quick Verification

After deployment, test:
1. Visit: https://arooon-n.github.io/literature-mining-hydrogen-storage-alloys/
2. Start backend: `python main_fastapi.py`
3. You should see "✅ System Ready • Model: gpt-oss:120b-cloud"
4. Upload a PDF and test extraction

## 📝 Troubleshooting

### Workflow fails?
- Check the Actions tab for error details
- Ensure GitHub Pages is enabled in Settings
- Verify the workflow file is in `.github/workflows/deploy.yml`

### Can't push to GitHub?
```powershell
# Check if remote is set
git remote -v

# If not set, add it:
git remote add origin https://github.com/arooon-n/literature-mining-hydrogen-storage-alloys.git
```

### Need to force push?
```powershell
git push -f origin main
```

## ✅ Success Checklist
- [ ] GitHub Pages enabled in repository settings
- [ ] Code pushed to main branch
- [ ] GitHub Actions workflow completed successfully
- [ ] Frontend accessible at GitHub Pages URL
- [ ] Backend running locally on port 8000
- [ ] Frontend successfully connects to backend
