#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <exception>

// 錯誤檢查巨集
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file,
           const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}

// 常量定義
#define TILE_SIZE 16
#define BLOCK_ROWS 8  // 每個線程處理多行來提高計算密度

__global__ void optimized_tiled_matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ bias,
    int N1, int N2, int N3
) {
    // 保持簡單的索引計算
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    // 直接計算全局索引
    int i = TILE_SIZE*by + ty;
    int j = TILE_SIZE*bx + tx;
    
    // 使用向量化載入優化記憶體存取
    float4* A4 = (float4*)A;
    float4* B4 = (float4*)B;
    
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE];
    
    float value = 0;
    
    #pragma unroll 4
    for (int phase = 0; phase < ceil((float)N2/TILE_SIZE); phase++) {
        // 優化的記憶體載入
        if ((i < N1) && ((phase*TILE_SIZE+tx) < N2)) {
            int idx = (i*N2 + phase*TILE_SIZE+tx)/4;
            float4 tmp = A4[idx];
            sh_A[ty][tx] = tmp.x;  // 只取需要的元素
        }
        
        __syncthreads();
        
        // 保持簡單的計算核心
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if ((i < N1) && (j < N3)) {
        C[i*N3+j] = value + bias[j];
    }
}

__global__ void tiled_mat_mul_kernel(float* A, float* B, float* C, float* bias, int N1, int N2, int N3)
{

    
    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int ty = threadIdx.y;
    int tx = threadIdx.x; 

    // Working on C[i,j]
    int i = TILE_SIZE*by + ty;
    int j = TILE_SIZE*bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < ceil((float)N2/TILE_SIZE); phase++)
    {
        // Load Tiles into shared memory
        if ((i < N1) && ((phase*TILE_SIZE+tx) < N2))
          sh_A[ty][tx] = A[(i)*N2 + phase*TILE_SIZE+tx];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_SIZE + ty) < N2) && (j < N3))
          sh_B[ty][tx] = B[(phase*TILE_SIZE + ty)*N3+j];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_SIZE; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((i < N1) && (j < N3))
      C[i*N3+j] = value + bias[j];
}

__global__ void matrixMultiply_broadcast(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ output, 
    const float* __restrict__ bias,
    int M, 
    int N, 
    int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int A_row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;

    // 計算需要處理的 tile 數量
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // 載入 A 和 B 矩陣的 tile 到共享記憶體
        int tileIdx = tile * TILE_SIZE;
        
        if (A_row < M && threadIdx.x < TILE_SIZE && (tileIdx + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[A_row * K + tileIdx + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && threadIdx.y < TILE_SIZE && (tileIdx + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tileIdx + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // 計算當前 tile 的部分和
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 寫入最終結果
    if (A_row < M && col < N) {
        output[A_row * N + col] = sum + bias[col];
    }
}

__global__ void matrixMultiply(const float* A, const float* B, float* C, const float* bias, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}

__global__ void matrixMultiply_broadcast_add(const float* A, const float* B, float* output, const float* bias, int M, int N, int K) {
    int A_row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (A_row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += (B[i * N + col] > 0 ? A[A_row * K + i] : -A[A_row * K + i]);
        }
        output[A_row * N + col] = sum + bias[col];
    }
}

// 矩陣乘法的主機端配置和調用
class MatrixMultiplier {
private:
    // static constexpr int TILE_SIZE = 32;
    // static constexpr int BLOCK_ROWS = 8;

    // 設備記憶體指標
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    float *d_C2 = nullptr;
    float *d_bias = nullptr;

    // 主機記憶體指標
    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    float *h_C2 = nullptr;
    float *h_bias = nullptr;

    // 矩陣維度
    int M, N, K;

public:
    MatrixMultiplier(int m, int n, int k) : M(m), N(n), K(k) {
        allocateMemory();
    }

    ~MatrixMultiplier() {
        freeMemory();
    }

    // 分配主機和設備記憶體
    void allocateMemory() {
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);
        size_t sizeBias = N * sizeof(float);

        // 分配主機記憶體
        h_A = (float*)malloc(sizeA);
        h_B = (float*)malloc(sizeB);
        h_C = (float*)malloc(sizeC);
        h_C2 = (float*)malloc(sizeC);
        h_bias = (float*)malloc(sizeBias);

        // 分配設備記憶體
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeA));
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeB));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, sizeC));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C2, sizeC));
        CHECK_CUDA_ERROR(cudaMalloc(&d_bias, sizeBias));
    }

    // 釋放記憶體
    void freeMemory() {
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C) free(h_C);
        if (h_C2) free(h_C2);
        if (h_bias) free(h_bias);

        if (d_A) CHECK_CUDA_ERROR(cudaFree(d_A));
        if (d_B) CHECK_CUDA_ERROR(cudaFree(d_B));
        if (d_C) CHECK_CUDA_ERROR(cudaFree(d_C));
        if (d_C2) CHECK_CUDA_ERROR(cudaFree(d_C2));
        if (d_bias) CHECK_CUDA_ERROR(cudaFree(d_bias));
    }

    // 初始化矩陣資料
    void initializeData() {
        // 初始化輸入矩陣
        for (int i = 0; i < M * K; i++) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            // h_A[i] = i;
        }
        for (int i = 0; i < K * N; i++) {
            // h_B[i] = static_cast<float>(rand()) / RAND_MAX;
            if (i % 2 == 0) {
                h_B[i] = 1;
            } else {
                h_B[i] = -1;
            }
            
        }
        // 初始化偏置
        for (int i = 0; i < N; i++) {
            h_bias[i] = static_cast<float>(rand()) / RAND_MAX;
            // h_bias[i] = 0.0f;
        }

        // 複製資料到設備
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), 
                                  cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), 
                                  cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, N * sizeof(float), 
                                  cudaMemcpyHostToDevice));
    }

    // 執行矩陣乘法
    void multiply() {
        // 配置網格和區塊維度
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + block.x - 1) / block.x, 
                 (M + (block.y) - 1) / (block.y));

        // 啟動核心
        
        matrixMultiply_broadcast<<<grid, block>>>(d_A, d_B, d_C2, d_bias, M, N, K);
        // optimized_tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, d_bias, M, K, N);
        // tiled_mat_mul_kernel<<<grid, block>>>(d_A, d_B, d_C, d_bias, M, N, K);
        matrixMultiply_broadcast_add<<<grid, block>>>(d_A, d_B, d_C, d_bias, M, N, K);
        // reset d_C
        // CHECK_CUDA_ERROR(cudaMemset(d_C, 0, M * N * sizeof(float)));
        // matrixMultiply<<<grid, block>>>(d_A, d_B, d_C, d_bias, M, N, K);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // 將結果複製回主機
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), 
                                  cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_C2, d_C2, M * N * sizeof(float), 
                                  cudaMemcpyDeviceToHost));
    }
    bool verify2kernelResult() {
        for(int i=0; i<=M*K; i++){
            if(h_C[i] != h_C2[i]){
                printf("i = %d, h_C[i] = %f, h_C2[i] = %f\n", i, h_C[i], h_C2[i]);
                return false;
            }
            // printf("i = %d, h_C[i] = %f, h_C2[i] = %f\n", i, h_C[i], h_C2[i]);
        }
        return true;
    }

    // 驗證結果
    bool verifyResult() {
        bool correct = true;
        // 在 CPU 上計算參考結果
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += h_A[i * K + k] * h_B[k * N + j];
                }
                sum += h_bias[j];
                
                // 比較結果，允許小誤差
                float diff = abs(h_C[i * N + j] - sum);
                if (diff > 1e-2) {
                    printf("Mismatch at position (%d, %d): CUDA=%f, CPU=%f\n", 
                           i, j, h_C[i * N + j], sum);
                    correct = false;
                    break;
                }
            }
        }
        return correct;
    }

    // 性能測試
    void benchmark(int iterations = 10) {
        // 預熱
        multiply();

        // 開始計時
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            multiply();
        }
        
        // 停止計時
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                       (end - start).count();
        
        float avgTime = duration / static_cast<float>(iterations);
        float gflops = (2.0f * M * N * K + M * N) * iterations / 
                      (duration * 1e-6) / 1e9;
        
        printf("Average time: %.3f ms\n", avgTime / 1000.0f);
        printf("Performance: %.2f GFLOPS\n", gflops);
    }
};

int main() {
    // 設定矩陣維度
    int M = 2048;  // A 矩陣的行數
    int N = 2048;  // B 矩陣的列數
    int K = 4096;  // A 矩陣的列數/B 矩陣的行數

    try {
        // 初始化 CUDA 設備
        int deviceId = 0;
        CHECK_CUDA_ERROR(cudaSetDevice(deviceId));

        // 創建矩陣乘法器實例
        MatrixMultiplier multiplier(M, N, K);

        // 初始化資料
        multiplier.initializeData();

        // 執行矩陣乘法
        // printf("Performing matrix multiplication...\n");
        for(int i=0; i<=10; i++) multiplier.multiply();

        // 驗證結果
        printf("Verifying results...\n");
        // if (multiplier.verify2kernelResult()) {
        //     printf("Results verified successfully!\n");
        // } else {
        //     printf("Results verification failed!\n");
        // }

        // 執行性能測試
        // printf("\nRunning performance benchmark...\n");
        // multiplier.benchmark();

    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        return 1;
    }

    return 0;
}