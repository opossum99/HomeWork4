
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>

#define N 256
#define TOL 1e-8
#define MAX_ITER 1000000

__global__ void jacobi_kernel(double* d_u, double* d_u_new, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Hello from %d\n", i*n + j);

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        d_u_new[i * n + j] = 0.25f * (d_u[(i - 1) * n + j] + d_u[(i + 1) * n + j] +
            d_u[i * n + (j - 1)] + d_u[i * n + (j + 1)]);
    }
}

void initialize(double* u, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == 0) {
                u[i * n + j] = 1.0f;  // Boundary conditions at y=1
            }
            else {
                u[i * n + j] = 0.0f;  // Initial guess and other boundary conditions
            }
        }
    }
}

double get_error(double* u, double* u_new, int n) {
    double error = 0.0f;
    for (int i = 0; i < n * n; i++) {
        error = fmaxf(error, fabsf(u_new[i] - u[i]));
    }
    return error;
}

int main() {
    int n = N;
    size_t size = n * n * sizeof(double);

    // Host memory allocation
    double* h_u = (double*)malloc(size);
    double* h_u_new = (double*)malloc(size);

    // Device memory allocation
    double* d_u, * d_u_new;
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_u_new, size);

    // Initialize host array
    initialize(h_u, n);

    // Copy to device
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, h_u, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int iter = 0;
    double error = 1.0f;

    // Main iteration loop
    while (error > TOL && iter < MAX_ITER) {
        jacobi_kernel <<<blocksPerGrid, threadsPerBlock >>> (d_u, d_u_new, n);
        cudaDeviceSynchronize();
        //printf("\n");

        // Swap pointers
        double* temp = d_u;
        d_u = d_u_new;
        d_u_new = temp;

        // Copy result to host to calculate error
        cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_u_new, d_u_new, size, cudaMemcpyDeviceToHost);

        error = get_error(h_u, h_u_new, n);
        iter++;
    }

    FILE* f;
    f = fopen("out.txt", "w");
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            fprintf(f, "%f ", h_u[i * n + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Error: " << error << std::endl;

    // Free memory
    free(h_u);
    free(h_u_new);
    cudaFree(d_u);
    cudaFree(d_u_new);

    return 0;
}
