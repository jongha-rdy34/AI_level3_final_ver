@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo [1/6] Python 3.12 확인
py -3.12 --version >nul 2>&1 || (echo Python 3.12가 없습니다. python.org에서 3.12를 설치하세요. & exit /b 1)
if not exist .venv ( py -3.12 -m venv .venv || exit /b 1 )
set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip
echo [2/6] torch 설치 (GPU 있으면 CUDA 12.8, 없으면 CPU)
where nvidia-smi >nul 2>&1 && (set TORCH_INDEX=https://download.pytorch.org/whl/cu128) || (set TORCH_INDEX=https://download.pytorch.org/whl/cpu)
echo     index: %TORCH_INDEX%
%PY% -m pip install torch==2.11.0 torchvision==0.26.0 --index-url %TORCH_INDEX% || exit /b 1
echo [3/6] 나머지 패키지
%PY% -m pip install -r requirements.txt || exit /b 1
%PY% -m ipykernel install --sys-prefix --name lv3venv --display-name "Python (.venv lv3)"
echo [4/6] 사전학습 가중치 (HuggingFace 공개 파일, 약 21MB)
if not exist tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors curl -L -o tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k/resolve/main/model.safetensors
echo [5/6] Kaggle 데이터 (공개 데이터셋, 로그인 없이 받음. 약 940MB)
if not exist raw mkdir raw
if not exist raw\fashion.zip curl -L -o raw\fashion.zip https://www.kaggle.com/api/v1/datasets/download/paramaggarwal/fashion-product-images-small
if not exist raw\utkface.zip curl -L -o raw\utkface.zip https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new
echo [6/6] 노트북 형식으로 변환 + P3 연결 파일
if not exist data\utkface\val.csv %PY% prepare_data.py || exit /b 1
if not exist gms_config.pkl %PY% make_gms_config.py
%PY% -c "import torch,timm;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),\"timm\",timm.__version__)"
echo 완료. VSCode에서 이 폴더를 열고 커널로 .venv\Scripts\python.exe 를 선택하세요.
endlocal
