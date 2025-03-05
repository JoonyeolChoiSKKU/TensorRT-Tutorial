#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstring>

// TensorRT Logger
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// 엔진 파일에서 엔진을 로드(deserialize)하는 함수
nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFilePath, nvinfer1::ILogger& logger)
{
    // 1) trt 파일을 바이너리 모드로 읽기
    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "ERROR: Could not open engine file: " << engineFilePath << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    long fsize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    file.close();

    // 2) IRuntime 생성
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime)
    {
        std::cerr << "ERROR: Could not create TensorRT runtime." << std::endl;
        return nullptr;
    }

    // 3) deserializeCudaEngine() 호출
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (!engine)
    {
        std::cerr << "ERROR: Failed to deserialize engine." << std::endl;
        delete runtime; // 8.5+ destroy() 대신 delete
        return nullptr;
    }

    // runtime는 엔진 생성 후 더 이상 필요 없으므로 해제
    delete runtime;
    return engine;
}

int main()
{
    // Logger 준비
    Logger logger;

    // 1. .trt 파일에서 엔진 로드
    std::string enginePath = "../simple_nn.trt";
    nvinfer1::ICudaEngine* engine = loadEngineFromFile(enginePath, logger);
    if (!engine)
    {
        std::cerr << "Engine load failed!" << std::endl;
        return -1;
    }
    std::cout << "Successfully loaded TensorRT engine: " << enginePath << std::endl;

    // 2. Execution Context 생성
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Failed to create ExecutionContext!" << std::endl;
        delete engine;
        return -1;
    }

    // 3. 동적 shape 설정
    //  - 파이썬 build 시 EXPLICIT_BATCH로 [1..8,10] 범위를 설정했으므로,
    //    여기서는 예시로 배치=1 => shape=[1,10]으로 세팅
    bool setShapeOk = context->setInputShape("input", nvinfer1::Dims2(1, 10));
    if (!setShapeOk)
    {
        std::cerr << "setInputShape failed for 'input'." << std::endl;
        delete context;
        delete engine;
        return -1;
    }

    // 4. 입력/출력 버퍼 준비
    int batchSize = 1;
    int inputSize = 10;  // simple_nn.onnx에서 Linear(10->20)이므로, 입력은 10
    int outputSize = 5;  // 최종 fc2(20->5)

    std::vector<float> hostInput(batchSize * inputSize);
    std::vector<float> hostOutput(batchSize * outputSize, 0.0f);

    // 입력값 채우기(예시)
    for (int i = 0; i < batchSize * inputSize; ++i)
    {
        hostInput[i] = float(i) * 0.1f;
    }

    // GPU 메모리 할당
    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;
    cudaMalloc(&deviceInput,  inputSize  * batchSize * sizeof(float));
    cudaMalloc(&deviceOutput, outputSize * batchSize * sizeof(float));

    cudaMemcpy(deviceInput, hostInput.data(),
               inputSize * batchSize * sizeof(float),
               cudaMemcpyHostToDevice);

    // 5. Named I/O API: 입력/출력 텐서에 GPU 버퍼 주소 지정
    context->setInputTensorAddress("input", deviceInput);
    context->setOutputTensorAddress("output", deviceOutput);

    // 6. 추론 실행
    // enqueueV3() 사용: stream=0(동기)
    cudaStream_t stream = nullptr;
    bool status = context->enqueueV3(stream);
    if (!status)
    {
        std::cerr << "Inference failed at enqueueV3!" << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        delete context;
        delete engine;
        return -1;
    }

    // 동기화
    cudaStreamSynchronize(stream);

    // 7. 결과를 CPU로 복사
    cudaMemcpy(hostOutput.data(), deviceOutput,
               outputSize * batchSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 결과 출력
    std::cout << "Inference result [batch=1, outputSize=5]: ";
    for (int i = 0; i < outputSize; ++i)
    {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;

    // 8. 리소스 정리
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    delete context;  // 8.5+ destroy() → delete
    delete engine;

    return 0;
}
