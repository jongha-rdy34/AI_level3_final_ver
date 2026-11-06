@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set S=..\AI_Lv3_sample_solution
if not exist %S%\setup.cmd (echo 옆 폴더 AI_Lv3_sample_solution 이 필요합니다. & exit /b 1)
echo [1/4] 샘플 폴더의 환경(.venv, 가중치, Kaggle 원본)을 먼저 준비합니다
if not exist %S%\data\utkface\val.csv call %S%\setup.cmd || exit /b 1
echo [2/4] .venv 연결 (샘플 폴더의 .venv를 같이 씀, 디스크 절약)
if not exist .venv mklink /J .venv %S%\.venv
if not exist tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors copy %S%\tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors .
echo [3/4] 예측 문제 데이터 생성
if not exist data\f2_2_kaggle\prepared.npz .venv\Scripts\python prepare_final.py || exit /b 1
echo [4/4] F3 파일과 연결 파일
.venv\Scripts\python _build\f3_data.py
if not exist gms_config.pkl .venv\Scripts\python make_gms_config.py
echo 완료. VSCode에서 이 폴더를 열고 커널로 .venv\Scripts\python.exe 를 선택하세요.
endlocal
