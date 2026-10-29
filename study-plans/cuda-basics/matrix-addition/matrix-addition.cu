#include <cuda_runtime.h>
// #include <stdio.h>


__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    // Write the float32 result into C; this kernel returns no value.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        // C[i][j] = A[i][j] + B[i][j];
        int flat = i * N + j;
        C[flat] = A[flat] + B[flat];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();

    // printf("last elemetn of A %f, B %f", A[4], B[4]);

    // printf("Blocks grid dimensions: x=%u, y=%u, z=%u\n", blocks.x, blocks.y, blocks.z);

}
