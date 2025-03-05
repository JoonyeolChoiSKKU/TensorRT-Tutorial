# TensorRT-Tutorial

이 저장소는 PyTorch 모델을 ONNX로 내보내고, TensorRT에서 다양한 방식(런타임 파싱 / 사전 변환)을 통해 C++ 추론을 수행하는 예시를 담고 있습니다.

---

## 1. TensorRT 설치 방법 (Ubuntu 예시)

1. **NVIDIA 레포지토리 추가**  
   - Ubuntu 22.04 기준, 다음과 같이 레포를 등록합니다. (버전에 따라 경로가 달라질 수 있습니다.)
     ```bash
     sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
     sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list'
     sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/tensorRT.list'
     sudo apt update
     ```

2. **TensorRT 패키지 설치**  
   ```bash
   sudo apt-get install tensorrt libnvinfer10 libnvinfer-dev python3-libnvinfer
```

- 필요한 경우, libnvinfer-plugin-dev, libnvinfer-bin, python3-libnvinfer-dev 등도 설치 가능합니다.
 
1. **버전 확인** 

```bash
dpkg -l | grep nvinfer
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

  - 여기서 Python과 C++(dpkg) 버전이 일치해야 엔진 호환 문제를 피할 수 있습니다.

## 2. 폴더 구조 


---

TensorRT-Tutorial
├─ post-trt/
│ ├─ CMakeLists.txt
│ ├─ main.cpp
│ └─ build/ (cmake 빌드 폴더)
└─ pre-trt/
├─ CMakeLists.txt
├─ main.cpp
├─ NN_to_ONNX.py (PyTorch 모델을 ONNX로 내보내는 코드)
├─ ONNX_to_TRT.py (ONNX를 .trt 엔진 파일로 변환)
├─ simple_nn.trt (예시 엔진 파일; 필요 시 .gitignore에서 제외)
├─ simple_nn.onnx (예시 ONNX 파일)
└─ build/ (cmake 빌드 폴더)
- post-trt: C++에서 ONNX를 직접 파싱하여 엔진을 만드는 구조

- pre-trt: 미리 .trt 엔진을 만들어두고, C++에서 로드(deserialize)하는 구조

## 3. 사용 방법 


---


### 3.1 NN_to_ONNX.py (PyTorch → ONNX) 
 
- 먼저 PyTorch 모델(.pt 등)을 읽어와서 ONNX 변환. 여기서는 예시로 NN_to_ONNX.py가 simple_nn.onnx를 생성한다.


```bash
cd TensorRT-Tutorial/pre-trt
python3 NN_to_ONNX.py
# simple_nn.onnx 생성 확인
```

### 3.2 pre-trt 폴더 (.trt 사전 생성 방식) 
 
1. ONNX_to_TRT.py 실행 (ONNX → .trt 엔진 변환)


```bash
cd TensorRT-Tutorial/pre-trt
python3 ONNX_to_TRT.py
# simple_nn.trt 파일 생성
```
 
2. CMake 빌드


```bash
mkdir build
cd build
cmake ..
make
```
 
3. 추론 실행


```bash
./run_trt
```

  - 여기서 C++ main.cpp는 simple_nn.trt를 로드(deserialize)한 뒤, 추론을 수행합니다.

### 3.3 post-trt 폴더 (런타임 ONNX 파싱 방식) 
 
1. 사전에 NN_to_ONNX.py로 simple_nn.onnx를 만들어둔다.
 
2. 빌드


```bash
cd TensorRT-Tutorial/post-trt
mkdir build
cd build
cmake ..
make
```
 
3. 실행


```bash
./run_trt
```

  - 여기서는 C++ 코드가 simple_nn.onnx를 직접 파싱 → TensorRT 엔진 빌드 → 추론 진행.

## 4. 주의 사항 


---

 
1. TensorRT 버전 호환
 
  - 엔진을 빌드할 때 사용한 TensorRT 버전과, 로드할 때 사용하는 버전이 다르면
"The engine plan file is not compatible …" 오류가 발생합니다.

  - Python tensorrt.__version__과 dpkg libnvinfer10 버전을 일치시키세요.
 
2. CUDA/GPU 드라이버 호환성

  - CUDA 11.x, 12.x 등 버전에 맞춰 TensorRT 버전을 설치하십시오.
 
3. .gitignore

  - 빌드 산출물, .trt, .onnx 파일 등을 버전 관리에서 제외하려면 적절히 .gitignore 파일을 설정하세요.

## 5. 참고 링크 


---

 
- NVIDIA TensorRT 공식 문서
[https://docs.nvidia.com/deeplearning/tensorrt/](https://docs.nvidia.com/deeplearning/tensorrt/)
 
- PyTorch → ONNX → TensorRT 예시
[https://github.com/NVIDIA/TensorRT/tree/main/quickstart](https://github.com/NVIDIA/TensorRT/tree/main/quickstart)
 
- trtexec 도움말
[https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)

