# Hydrogen Storage Alloys - LLM Literature Mining

## 🚀 Live Demo
**Backend**: You need to run the backend locally (instructions below)

## 📋 Prerequisites
- Python 3.8+
- Ollama running locally with `gpt-oss:120b-cloud` model
- Git

## 🛠️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/arooon-n/literature-mining-hydrogen-storage-alloys.git
cd literature-mining-hydrogen-storage-alloys
```

### 2. Install Python dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Start the backend server
```bash
python main_fastapi.py
```

The backend will start at: http://localhost:8000

### 4. Access the application
- Open: http://localhost:8000/

## 🌐 GitHub Pages Deployment

The frontend is automatically deployed to GitHub Pages when you push to the `main` branch.

### Manual deployment (if needed):
```bash
# Push your changes
git add .
git commit -m "Update frontend"
git push origin main
```

GitHub Actions will automatically:
1. Build and deploy the frontend to GitHub Pages
2. Configure it to connect to http://localhost:8000 for API calls

## 📝 Configuration

### Environment Variables (Optional)
```bash
# Windows PowerShell
$env:OLLAMA_URL="http://localhost:11434"
$env:MODEL_NAME="gpt-oss:120b-cloud"

# Linux/Mac
export OLLAMA_URL="http://localhost:11434"
export MODEL_NAME="gpt-oss:120b-cloud"
```

## 📂 Project Structure
```
.
├── main_fastapi.py          # Backend API server
├── llm_extractor.py          # LLM extraction logic
├── requirements.txt          # Python dependencies
├── frontend/                 # Frontend files (deployed to GitHub Pages)
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── data/                     # Data directory
│   ├── pdfs/
│   └── raw_text/
├── outputs/                  # Extracted CSV files
└── uploads/                  # Uploaded PDFs

```

## 📊 Features
- PDF upload and text extraction
- Multi-chunk LLM processing for large papers
- Alloy data extraction with AI model
- CSV export of extracted data
- Real-time progress tracking
- Loading overlays during processing

## 👥 Team
AIE - B | Group - 17
