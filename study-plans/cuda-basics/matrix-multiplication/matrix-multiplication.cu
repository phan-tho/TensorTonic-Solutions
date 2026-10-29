#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Write code here
    // A shape = M, K
    // B shape = K N
    // C shape = M N

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    // hang i cot j
    
    if (j < N && i < M) {
        int idx = i * N + j;
        C[idx] = 0;
        for (int k = 0; k < K; k++) {
            C[idx] += A[i * K + k] * B[k * N + j];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
