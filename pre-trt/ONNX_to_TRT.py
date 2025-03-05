import tensorrt as trt
import numpy as np

print("Python tensorrt.__version__:", trt.__version__)

# ONNX 파일 경로
onnx_path = "../simple_nn.onnx"
trt_path = "simple_nn.trt"

# TensorRT 로그 생성
logger = trt.Logger(trt.Logger.WARNING)

# ONNX 모델을 TensorRT 엔진으로 변환하는 함수
def build_engine(onnx_file_path):
    with trt.Builder(logger) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser, \
         builder.create_builder_config() as config:
        
        config.set_flag(trt.BuilderFlag.FP16)  # FP16 모드 사용 가능 (원하지 않으면 주석 처리)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1MB 메모리 워크스페이스 설정

        # ONNX 모델을 읽어와서 TensorRT 변환
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # 최적화 프로파일 추가 (동적 크기 지원)
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)  # 첫 번째 입력 텐서 가져오기

        min_shape = (1, 10)   # 최소 배치 크기 1
        opt_shape = (4, 10)   # 최적 배치 크기 4
        max_shape = (8, 10)   # 최대 배치 크기 8

        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # TensorRT 8.5+에서는 buildSerializedNetwork()를 사용하여 직렬화된 엔진 생성
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("❌ 엔진 직렬화 실패!")
            return None

        # 실행 가능한 엔진으로 변환
        runtime = trt.Runtime(logger) 
        return runtime.deserialize_cuda_engine(serialized_engine)

# TensorRT 엔진 빌드
engine = build_engine(onnx_path)
if engine:
    with open(trt_path, "wb") as f:
        f.write(engine.serialize())
    print(f"✅ TensorRT 엔진이 {trt_path}로 저장되었습니다.")
else:
    print("❌ TensorRT 변환에 실패했습니다.")
