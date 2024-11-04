#include <stdio.h>
#include <iostream>

// 可以在 kernel 中被存取的全局內存靜態變量 __device__
__device__ float d_x = 8;

__device__ float add(const float a, const float b) {
    return a + b;
}

__global__ void hello_from_gpu(float *a, float *b, float *c) {
    
    const int bid = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    const int id = bid * (blockDim.x * blockDim.y) + tid;

    c[id] = add(a[id], b[id]) + d_x;
    
    // printf("Hello from block %d, thread %d, id %d\n", bid, tid, id);
    // printf("c[%d] = %f + %f\n", id, a[id], b[id]);


}

int main(void){

    
    int target = 64;
    dim3 grid_size(target / 32);
    dim3 block_size(32);
    size_t byte_count = sizeof(float) * target;

    float *fpHost_a, *fpHost_b, *fpHost_c;


    fpHost_a = (float *)malloc(byte_count);
    fpHost_b = (float *)malloc(byte_count);
    fpHost_c = (float *)malloc(byte_count); 

    memset(fpHost_a, 0, byte_count);
    memset(fpHost_b, 0, byte_count);
    memset(fpHost_c, 0, byte_count);

    for(int i=0; i<target; i++){
        fpHost_a[i] = i;
        fpHost_b[i] = i;
        fpHost_c[i] = i;
    }

    float *fpDevice_a;
    float *fpDevice_b;
    float *fpDevice_c;

    cudaMalloc((float **)&fpDevice_a, byte_count);
    cudaMalloc((float **)&fpDevice_b, byte_count);
    cudaMalloc((float **)&fpDevice_c, byte_count);

    cudaMemset(fpDevice_a, 0, byte_count);
    cudaMemset(fpDevice_b, 0, byte_count);
    cudaMemset(fpDevice_c, 0, byte_count);

    cudaMemcpy(fpDevice_a, fpHost_a, byte_count, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_b, fpHost_b, byte_count, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDevice_c, fpHost_c, byte_count, cudaMemcpyHostToDevice);

    hello_from_gpu<<<grid_size, block_size>>>(fpDevice_a, fpDevice_b, fpDevice_c);

    cudaMemcpy(fpHost_c, fpDevice_c, byte_count, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++){
        printf("i = %d, in c[i] = %f\n", i, fpHost_c[i]);
    }

    cudaFree(fpDevice_a);
    cudaFree(fpDevice_b);
    cudaFree(fpDevice_c);

    free(fpHost_a);
    free(fpHost_b);
    free(fpHost_c);

    
    cudaDeviceSynchronize();
    
    
    return 0;
}