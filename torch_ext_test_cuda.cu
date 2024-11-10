#include <torch/extension.h>
#include <iostream>

// #include "cublas_v2.h"

const unsigned int TILE_WIDTH = 32;

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
            sum += A[A_row * K + i] * B[i * N + col];
            // printf("row: %d, col: %d, col[i]: %d, sum: %f\n", A_row, col, i, sum);
        }
        output[A_row * N + col] = sum + bias[col];
    }
}

__global__ void matrixMultiply_broadcast_add(const float* A, const float* B, float* output, const float* bias, int M, int N, int K) {
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
            // sum += A[A_row * K + i] * B[i * N + col];
            if(B[i * N + col] == -1){
                sum -= A[A_row * K + i];
            } else {
                sum += A[A_row * K + i];
            }
            // printf("row: %d, col: %d, col[i]: %d, sum: %f\n", A_row, col, i, sum);
        }
        output[A_row * N + col] = sum + bias[col];
    }
}

__global__ void matrixMultiply_broadcast_qkv(
    const float* q, const float* w_q, const float* b_q, 
    const float* k, const float* w_k, const float* b_k, 
    const float* v, const float* w_v, const float* b_v, 
    float* output_q, float* output_k, float* output_v,
    int M, int N, int K,
    int M_q, int M_k, int M_v) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int B_row = (blockIdx.y * blockDim.y + threadIdx.y) % K;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M_q) {
        if (col < N) {
            float sum = 0;
            for (int i = 0; i < K; ++i) {
                sum += q[row * K + i] * w_q[i * N + col];
            }
            output_q[row * N + col] = sum + b_q[col];
        }
    } else if(row < M_q + M_k) {
        if (col < N) {
            float sum = 0;
            for (int i = 0; i < K; ++i) {
                sum += k[(row - M_q) * K + i] * w_k[i * N + col];
            }
            output_k[(row - M_q) * N + col] = sum + b_k[col];
        }
    } else {
        if (col < N) {
            float sum = 0;
            for (int i = 0; i < K; ++i) {
                sum += v[(row - M_q - M_k) * K + i] * w_v[i * N + col];
            }
            output_v[(row - M_q - M_k) * N + col] = sum + b_v[col];
        }
    }

    // if (A_row < M && col < N) {
    //     float sum = 0;
    //     for (int i = 0; i < K; ++i) {
    //         sum += A[A_row * K + i] * B[i * N + col];
    //     }
    //     output[A_row * N + col] = sum + bias[col];
    // }
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
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x + 1, (M + block.y - 1) / block.y + 1);

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

    // ========================================
    // Cublas 矩陣乘法計算 (沒比較快)
    // cublasHandle_t handle;
    // cublasStatus_t status = cublasCreate(&handle);

    // if (status != CUBLAS_STATUS_SUCCESS) {
    //     throw std::runtime_error("CUBLAS initialization failed");
    // }

    // float alpha = 1.0f;
    // float beta = 0.0f;

    // status = cublasSgemm(
    //     handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, 
    //     (float *)b.t().contiguous().data_ptr(), N, (float *)a.data_ptr(), K, &beta, (float *)output.data_ptr(), N
    // );
    // ========================================

    matrixMultiply_broadcast<<<grid, block>>>(
        a.data_ptr<float>(), b.t().contiguous().data_ptr<float>(), output.data_ptr<float>(), bias.data_ptr<float>(), M, N, K
    );
    
    
    return output;
}

at::Tensor my_mm_bc_add(
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
    auto output = torch::zeros(new_sizes, bias.options()); // bias dtype = float
    // printf("M: %d, N: %d, K: %d\n", M, N, K);

    matrixMultiply_broadcast_add<<<grid, block>>>(
        a.data_ptr<float>(), b.t().contiguous().data_ptr<float>(), output.data_ptr<float>(), bias.data_ptr<float>(), M, N, K
    );
    
    
    return output;
}

at::Tensor create_output_tensor(const at::Tensor& a, const at::Tensor& b) {
    auto a_sizes = a.sizes().slice(0, a.sizes().size() - 1);
    std::vector<int64_t> new_sizes(a_sizes.begin(), a_sizes.end());
    new_sizes.push_back(b.size(0));
    return torch::zeros(new_sizes, a.options());
}

int cal_dim_before_last(const at::Tensor& t) {
    int n = 1;
    for(auto s : t.sizes()) {
        n *= s;
    }
    return n / t.size(-1);
}

std::vector<at::Tensor> my_mm_qkv(
    const at::Tensor& q,
    const at::Tensor& w_q, 
    const at::Tensor& b_q,
    const at::Tensor& k,
    const at::Tensor& w_k,
    const at::Tensor& b_k,
    const at::Tensor& v,
    const at::Tensor& w_v,
    const at::Tensor& b_v
) {
    // 檢查 input 的最後一維是否與 weight 的第一維相等
    if (q.size(-1) != w_q.size(1) || k.size(-1) != w_k.size(1) || v.size(-1) != w_v.size(1)) {
        throw std::invalid_argument("The last dim of input must be equal to the first dim of weight");
    }

    int M_q = cal_dim_before_last(q);
    int M_k = cal_dim_before_last(k);
    int M_v = cal_dim_before_last(v);

    int K_q = q.size(-1);
    int K_k = k.size(-1);
    int K_v = v.size(-1);

    int N_q = w_q.size(0);
    int N_k = w_k.size(0);
    int N_v = w_v.size(0);
    
    // input 的維度為 [..., K]
    // 將 input 維度化為 [M, K]
    // weight 的維度為 [K, N]
    int total_M = M_q + M_k + M_v;
    int total_K = K_q;
    int total_N = N_q;

    dim3 block(16, 16);
    dim3 grid((total_N + block.x - 1) / block.x, (total_M + block.y - 1) / block.y);

    auto output_q = create_output_tensor(q, w_q);
    auto output_k = create_output_tensor(k, w_k);
    auto output_v = create_output_tensor(v, w_v);

    matrixMultiply_broadcast_qkv<<<grid, block>>>(
        q.data_ptr<float>(), w_q.t().contiguous().data_ptr<float>(), b_q.data_ptr<float>(), 
        k.data_ptr<float>(), w_k.t().contiguous().data_ptr<float>(), b_k.data_ptr<float>(),
        v.data_ptr<float>(), w_v.t().contiguous().data_ptr<float>(), b_v.data_ptr<float>(),
        output_q.data_ptr<float>(), output_k.data_ptr<float>(), output_v.data_ptr<float>(),
        total_M, total_N, total_K, M_q, M_k, M_v
    );

    // return output_q, output_k, output_v;
    return std::vector<at::Tensor>{output_q, output_k, output_v};
}