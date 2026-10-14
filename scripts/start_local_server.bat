@echo off
REM 启动本地评测后端：Qwen3-4B-Instruct-2507 Q4_K_M，GPU 全量 offload（Vulkan）
REM 需要 6GB+ 显存；如显存不足可把 -ngl 99 降到 32（部分 offload）
cd /d %~dp0..
.tools\llama.cpp\llama-server.exe ^
  -m .models\qwen3-4b-instruct-2507-q4km.gguf ^
  -ngl 99 -c 24576 -fa on -ctk q8_0 -ctv q8_0 ^
  --host 127.0.0.1 --port 8321 ^
  --threads 6 ^
  --alias qwen3-4b-instruct-2507
