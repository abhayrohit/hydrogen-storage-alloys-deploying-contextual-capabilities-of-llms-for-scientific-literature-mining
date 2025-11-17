# Cleanup Script - Remove unnecessary files from git repository

Write-Host "🧹 Cleaning up unnecessary files from repository..." -ForegroundColor Cyan

# Remove outputs
Write-Host "Removing output CSV files..." -ForegroundColor Yellow
git rm --cached outputs/*.csv 2>$null
if ($?) { Write-Host "✓ Removed outputs/*.csv" -ForegroundColor Green }

# Remove uploads
Write-Host "Removing uploaded PDF files..." -ForegroundColor Yellow
git rm --cached uploads/*.pdf 2>$null
if ($?) { Write-Host "✓ Removed uploads/*.pdf" -ForegroundColor Green }

# Remove raw text
Write-Host "Removing extracted text files..." -ForegroundColor Yellow
git rm --cached data/raw_text/*.txt 2>$null
if ($?) { Write-Host "✓ Removed data/raw_text/*.txt" -ForegroundColor Green }

# Remove Python cache
Write-Host "Removing Python cache..." -ForegroundColor Yellow
git rm -r --cached __pycache__ 2>$null
if ($?) { Write-Host "✓ Removed __pycache__/" -ForegroundColor Green }

# Remove backup files
Write-Host "Removing backup files..." -ForegroundColor Yellow
git rm --cached *_backup.js 2>$null
git rm --cached *_clean.js 2>$null
git rm --cached frontend/*_backup.js 2>$null
if ($?) { Write-Host "✓ Removed backup files" -ForegroundColor Green }

Write-Host ""
Write-Host "📝 Files removed from git tracking (but kept locally)" -ForegroundColor Cyan
Write-Host "Now commit and push the changes:" -ForegroundColor White
Write-Host ""
Write-Host "  git commit -m 'Remove unnecessary files from repository'" -ForegroundColor Gray
Write-Host "  git push origin main" -ForegroundColor Gray
Write-Host ""
