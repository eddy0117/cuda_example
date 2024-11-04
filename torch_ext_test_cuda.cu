#include <torch/extension.h>
#include <iostream>


__global__ void add(float *a, float *b, float *output, int n) {
    
    const int bid = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    const int id = bid * (blockDim.x * blockDim.y) + tid;

    if (id >= n) {
        return;
    }

    output[id] = a[id] + b[id];

}

at::Tensor my_add(const torch::Tensor &a, 
             const torch::Tensor &b) {

    if (a.sizes() != b.sizes()) {
        throw std::invalid_argument("a and b must have the same size");
    }
    
    // torch::Tensor a_reshaped = a.reshape({-1});
    // torch::Tensor b_reshaped = b.reshape({-1});
    int n = 1;
    for(auto s : a.sizes()) {
        n *= s;
    }
    // int n = a_reshaped.size(0);
    const int sqrt_n = sqrt(n);
    const dim3 grid(n / 256 + 1);
    
    const dim3 block(16, 16);

    // std::cout << "flatten" << std::endl;
    // std::cout << a_reshaped << std::endl;

    // std::cout << "reshaped to original shape" << std::endl;
    // std::cout << a_reshaped.reshape(a.sizes()) << std::endl;

    // std::cout << a.sizes() << std::endl;
    torch::Tensor output = torch::zeros_like(a, a.options());
    
    
    add<<<grid, block>>>((float *)a.data_ptr(), (float *)b.data_ptr(), (float *)output.data_ptr(), n);
    
    return output;

}

__global__ void matrixMultiply(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMultiply_broadcast(const float* A, const float* B, float* output, float* bias, int M, int N, int K) {
    int A_row = blockIdx.y * blockDim.y + threadIdx.y;
    // int B_row = (blockIdx.y * blockDim.y + threadIdx.y) % K;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("A_row: %d, col: %d\n", A_row, col);
    if (A_row < M && col < N) {
        // for(int i = 0; i < (N * K); i++) {
        //     printf("B[%d] = %f\n", i, B[i]);
        // }
        float sum = 0;
        for (int i = 0; i < K; ++i) {
            sum += A[A_row * K + i] * B[col * N + i];
            // printf("row: %d, col: %d, col[i]: %d, sum: %f\n", A_row, col, i, sum);
        }
        output[A_row * N + col] = sum + bias[col];
    }
}

at::Tensor my_mm(
    const at::Tensor& a, 
    const at::Tensor& b
) {

    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    // 檢查 a 的列數是否等於 b 的行數
    if (K != b.size(0)) {
        throw std::invalid_argument("Matrices have incompatible dimensions for multiplication");
    }

    // 初始化輸出張量 c
    auto c = torch::zeros({M, N}, a.options());

    // 定義 CUDA block 和 grid 大小
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 呼叫 CUDA kernel 進行矩陣乘法
    matrixMultiply<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K
    );

    return c;
}

at::Tensor my_mm_bc(
    const at::Tensor& a, 
    const at::Tensor& b,
    const at::Tensor& bias
) {
    
    // auto b_t = b.transpose(0, 1);
    // 檢查 input 的最後一維是否與 weight 的第一維相等
    if (a.size(-1) != b.size(1)) {
        throw std::invalid_argument("The last dim of input must be equal to the first dim of weight");
    }

    int n = 1;
    for(auto s : a.sizes()) {
        n *= s;
    }
    
    // input 的維度為 [..., K]
    // 將 input 維度化為 [M, K]
    // weight 的維度為 [K, N]

    int M = n / a.size(-1);
    int K = a.size(-1);
    int N = b.size(0);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 輸出維度為 [..., N]
    auto a_sizes = a.sizes().slice(0, a.sizes().size() - 1);
    std::vector<int64_t> new_sizes(a_sizes.begin(), a_sizes.end());
    new_sizes.push_back(N);
    
    auto output = torch::zeros(new_sizes, a.options());
    // printf("M: %d, N: %d, K: %d\n", M, N, K);
    // std::cout << "b\n" << b << std::endl;
    // std::cout << "b_t\n" << b_t << std::endl;

    


    matrixMultiply_broadcast<<<grid, block>>>(
        a.data_ptr<float>(), b.t().data_ptr<float>(), output.data_ptr<float>(), bias.data_ptr<float>(), M, N, K
    );
    
    return output;
}