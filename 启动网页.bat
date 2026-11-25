@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv --system-site-packages .venv
    if errorlevel 1 goto :failed
)
.venv\Scripts\python.exe -c "import flask, ultralytics, torch, PIL, dill, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo 正在安装运行依赖，首次安装可能需要几分钟...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)
echo.
echo 本机访问地址：http://127.0.0.1:5000
echo 同一局域网的其他电脑请使用本机 IPv4 地址访问，例如：http://192.168.1.10:5000
echo 如需修改监听地址或端口，可在启动前设置 HOST 和 PORT 环境变量。
echo 按 Ctrl+C 停止服务。
.venv\Scripts\python.exe app.py
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo 启动失败，请查看上方错误信息。
pause
exit /b 1
