$ErrorActionPreference = "Stop"

$JETSON_USER = "raycast"
$JETSON_HOST = "192.168.55.1"
$REMOTE_DIR  = "/home/$JETSON_USER/raycast-app"
$SERVICE     = "raycast.service"

$LOCAL_ARCHIVE = Join-Path $env:TEMP "deploy_raycast.tgz"
if (Test-Path $LOCAL_ARCHIVE) { Remove-Item $LOCAL_ARCHIVE -Force }

tar -czf $LOCAL_ARCHIVE `
  --exclude=".git" `
  --exclude=".venv" `
  --exclude="__pycache__" `
  --exclude="*.pyc" `
  --exclude="deploy.tgz" `
  --exclude="*.env" `
  --exclude=".DS_Store" `
  --exclude="*.log" `
  --exclude="*.png" `
  .

Write-Host "=== Uploading archive to Jetson: /tmp/deploy.tgz ==="
scp $LOCAL_ARCHIVE "$JETSON_USER@${JETSON_HOST}:/tmp/deploy.tgz"

Write-Host "=== Running remote deploy script (remote output below) ==="

$remoteScript = @'
set -euxo pipefail

echo "=== Remote host: $(hostname) ==="
echo "=== deploy.tgz stats ==="
ls -lh /tmp/deploy.tgz

echo "=== deploy.tgz contents (first 40) ==="
tar -tzf /tmp/deploy.tgz | head -n 40 || true

echo "=== Stopping service (if running) ==="
systemctl --user stop __SERVICE__ || true

echo "=== Ensure dir exists: __REMOTE_DIR__ ==="
mkdir -p "__REMOTE_DIR__"

echo "=== BEFORE: __REMOTE_DIR__ ==="
ls -la "__REMOTE_DIR__" | head -n 80 || true

echo "=== Manual remove old code paths ==="
rm -rf "__REMOTE_DIR__/src" \
       "__REMOTE_DIR__/README.md" \
       "__REMOTE_DIR__/requirements.txt" \
       "__REMOTE_DIR__/runs" \
       "__REMOTE_DIR__/LICENSE" \
       "__REMOTE_DIR__/.vscode" \
       "__REMOTE_DIR__/.crossnote" \
       "__REMOTE_DIR__/deploy.ps1" \
       "__REMOTE_DIR__/requirements.local.lock" \
       "__REMOTE_DIR__/.python-version" \
       "__REMOTE_DIR__/.gitignore" \
       "__REMOTE_DIR__/.gitattributes" || true

echo "=== Extracting /tmp/deploy.tgz into __REMOTE_DIR__ ==="
tar -xzf /tmp/deploy.tgz -C "__REMOTE_DIR__" -o

echo "=== AFTER: __REMOTE_DIR__ ==="
ls -la "__REMOTE_DIR__" | head -n 80

echo "=== AFTER: __REMOTE_DIR__/src ==="
ls -la "__REMOTE_DIR__/src" | head -n 80

echo "=== Verify entrypoint ==="
test -f "__REMOTE_DIR__/src/main.py"

cd "__REMOTE_DIR__"
echo "=== Force correct venv (must be Python 3.10 on JP6.2) ==="
PY=python3.10
$PY --version

VENV_DIR=".venv"

# If venv exists, check what python it actually contains
if [ -x "$VENV_DIR/bin/python" ]; then
  VENV_PYVER="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' || echo "unknown")"
else
  VENV_PYVER="missing"
fi

echo "Existing venv python: $VENV_PYVER"

# Recreate if not 3.10
if [ "$VENV_PYVER" != "3.10" ]; then
  echo "Recreating venv with python3.10..."
  rm -rf "$VENV_DIR"
  $PY -m venv --system-site-packages "$VENV_DIR"
fi

. "$VENV_DIR/bin/activate"

echo "Venv python now: $(python -c 'import sys; print(sys.version)')"

echo "=== Pip bootstrap ==="
python -m pip install -U pip setuptools wheel

echo "=== Install JetPack 6.2 CUDA PyTorch (cp310 / cu126) ==="
python -m pip uninstall -y torch torchvision torchaudio || true

python -m pip install \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch torchvision torchaudio

echo "=== Verify CUDA torch ==="
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('is_available', torch.cuda.is_available()); print('count', torch.cuda.device_count())"

echo "=== Install the rest of deps (excluding torch/torchvision/torchaudio) ==="
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"

# Build a temp requirements file without torch pins so we don't overwrite CUDA torch
grep -vE '^(torch|torchvision|torchaudio)=' requirements.txt > /tmp/requirements.no_torch.txt

NEW_HASH="$(sha256sum /tmp/requirements.no_torch.txt | awk '{print $1}')"
OLD_HASH="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$NEW_HASH" = "$OLD_HASH" ]; then
  echo "=== requirements unchanged; skipping pip install ==="
else
  python -m pip install -r /tmp/requirements.no_torch.txt
  echo "$NEW_HASH" > "$REQ_HASH_FILE"
fi

# Safeguard installs
pip install filterpy==1.4.5
pip install depthai==2.31.0.0

# Additional Jetson-specific installs
pip install ultralytics[export]
# pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
# pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
# pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

python -m pip freeze > requirements.jetson.lock

echo "=== Starting service ==="
systemctl --user daemon-reload
systemctl --user start __SERVICE__
systemctl --user status __SERVICE__ --no-pager || true

echo "DEPLOY_OK"
deactivate
exit 0
'@

$remoteScript = $remoteScript.Replace("__REMOTE_DIR__", $REMOTE_DIR).Replace("__SERVICE__", $SERVICE)
$remoteScript = $remoteScript -replace "`r`n", "`n"

$remoteScript | ssh -tt "$JETSON_USER@$JETSON_HOST" "bash -s"
if ($LASTEXITCODE -ne 0) { throw "Remote deploy failed with exit code $LASTEXITCODE" }

Write-Host "Deploy complete."