# YOLOFlow Studio - dev launcher (Windows PowerShell)
# Starts the FastAPI backend on :8000 and the Vite dev server on :5173.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "$root\frontend\node_modules")) {
    Push-Location "$root\frontend"; npm install; Pop-Location
}
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; python -m uvicorn main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Start-Sleep -Seconds 4
Start-Process "http://localhost:5173"
