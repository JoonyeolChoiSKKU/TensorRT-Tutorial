#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

class Logger : public ILogger
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

int main()
{
    const std::string onnxModelPath = "../../simple_nn.onnx"; // PyTorch->ONNX 변환한 모델
    Logger logger;

    // (1) Builder, Network, Parser 준비
    IBuilder* builder = createInferBuilder(logger);
    if (!builder)
    {
        std::cerr << "Failed to create IBuilder!" << std::endl;
        return -1;
    }
    // TensorRT 10.x에서는 network creation flag (EXPLICIT_BATCH 등) 거의 deprecated.
    // 그냥 기본값(0)으로 network 생성
    INetworkDefinition* network = builder->createNetworkV2(0);
    if (!network)
    {
        std::cerr << "Failed to create INetworkDefinition!" << std::endl;
        delete builder;
        return -1;
    }

    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser)
    {
        std::cerr << "Failed to create ONNX parser!" << std::endl;
        delete network;
        delete builder;
        return -1;
    }

    // (2) ONNX 모델 로딩
    std::ifstream ifs(onnxModelPath, std::ios::binary);
    if (!ifs.good())
    {
        std::cerr << "ERROR: Could not open " << onnxModelPath << std::endl;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }
    std::string onnxContent((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    if (!parser->parse(onnxContent.data(), onnxContent.size()))
    {
        std::cerr << "ERROR: Failed to parse ONNX model." << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // (3) config 및 최적화 프로파일
    IBuilderConfig* config = builder->createBuilderConfig();
    if (!config)
    {
        std::cerr << "Failed to create IBuilderConfig!" << std::endl;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // 예시: FP16 모드 등
    // config->setFlag(BuilderFlag::kFP16);

    // workspace 제한
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 20); // 1MB

    // 텐서 이름이 "input"으로, shape [N, 10] (simple_nn.onnx)
    // 동적 배치: [1,10] ~ [8,10] 범위 지정
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims2(1, 10));
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims2(4, 10));
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims2(8, 10));
    config->addOptimizationProfile(profile);

    // (4) 엔진 빌드
    IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine)
    {
        std::cerr << "Failed to build serialized engine!" << std::endl;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // builder 관련 해제
    delete config;
    delete parser;
    delete network;
    delete builder;

    // runtime 생성
    IRuntime* runtime = createInferRuntime(logger);
    if (!runtime)
    {
        std::cerr << "Failed to create IRuntime!" << std::endl;
        // IHostMemory 해제
        delete serializedEngine; // destroy() 대신 delete
        return -1;
    }

    // 엔진 역직렬화
    ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine)
    {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        delete serializedEngine;
        delete runtime;
        return -1;
    }

    // IHostMemory 해제
    delete serializedEngine;

    std::cout << "Engine built from " << onnxModelPath << " successfully." << std::endl;

    // (5) Execution Context 생성
    IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Failed to create IExecutionContext!" << std::endl;
        delete engine;
        delete runtime;
        return -1;
    }

    // (6) Dynamic shape 세팅: batch=1 => [1, 10]
    // Named I/O API: setInputShape("tensorName", dims)
    bool shapeOk = context->setInputShape("input", Dims2(1, 10));
    if (!shapeOk)
    {
        std::cerr << "ERROR: setInputShape failed for 'input'." << std::endl;
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }

    // (7) 입력/출력 버퍼 준비
    int batchSize = 1;
    int inputSize = 10;
    int outputSize = 5;

    std::vector<float> hostInput(batchSize * inputSize, 0.1f);
    std::vector<float> hostOutput(batchSize * outputSize, 0.f);

    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;
    cudaMalloc(&deviceInput, batchSize * inputSize * sizeof(float));
    cudaMalloc(&deviceOutput, batchSize * outputSize * sizeof(float));

    cudaMemcpy(deviceInput, hostInput.data(),
               batchSize * inputSize * sizeof(float),
               cudaMemcpyHostToDevice);

    // (8) Named I/O API를 통해 텐서 주소 지정
    // 기존 bindingIndex 배열 대신, tensor 이름으로 세팅
    context->setInputTensorAddress("input", deviceInput);
    context->setOutputTensorAddress("output", deviceOutput);

    // (9) 추론 실행
    // enqueueV3(stream) 형태로 실행.
    // 여기서는 간단히 stream=0 (동기 실행)
    cudaStream_t stream = nullptr; // 0
    bool status = context->enqueueV3(stream);
    if (!status)
    {
        std::cerr << "Inference failed with enqueueV3!" << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }

    // 엔진 실행이 끝나길 기다리려면 cudaStreamSynchronize 필요
    cudaStreamSynchronize(stream);

    // (10) 결과 복사 및 출력
    cudaMemcpy(hostOutput.data(), deviceOutput,
               batchSize * outputSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "Inference result: ";
    for (int i = 0; i < outputSize; i++)
        std::cout << hostOutput[i] << " ";
    std::cout << std::endl;

    // (11) 리소스 정리
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    delete context; // destroy() 대신 delete
    delete engine;
    delete runtime;

    return 0;
}
