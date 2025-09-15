# 아나콘다 환경 접속 메뉴얼

### 아나콘다 환경 생성
```
conda create -n [환경이름] python=[파이썬버전] -y
```

### 아나콘다 환경 목록 확인
```
conda env list
```

### 아나콘다 환경 접속 명령어
```
- conda init

- conda activate tensor_gpu
```

### 아나콘다 최신 버전 업데이트
```
conda update conda
```

### 아나콘다 환경 접속 나가기 명령
```
conda deactivate
```

### 지정 아나콘다 환경 자동 실행 .vscode/settings.json
```
{
  "terminal.integrated.profiles.windows": {
    "Anaconda TensorGPU": {
      "source": "PowerShell",
      "args": [
        "-NoExit",
        "-Command",
        "conda activate tensor_gpu"
      ]
    }
  },
  "terminal.integrated.defaultProfile.windows": "Anaconda TensorGPU"
}
```

### 텐서 플로 설치
```
pip install tensorflow==2.10.0
```