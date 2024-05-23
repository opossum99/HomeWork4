#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)
#define WINDOW_SIZE 5

__device__ void bubble_sort(unsigned char* arr, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (arr[i] < arr[j]) {
                SWAP(arr[i], arr[j], double);
            }
        }
    }
}

__global__ void MedianKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    int halfWindowSize = WINDOW_SIZE / 2;

    if (x < width && y < height) {
        unsigned char window[WINDOW_SIZE * WINDOW_SIZE];
        //for (int c = 0; c < channels; c++) {
            int cnt = 0;
            for (int ky = -halfWindowSize; ky <= halfWindowSize; ky++) {
                for (int kx = -halfWindowSize; kx <= halfWindowSize; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;
                    if (0 <= ny < height && 0 <= nx < width) {
                        window[cnt++] = input[(ny * width + nx) * channels + c];
                    }
                }
            }
            bubble_sort(window, cnt);
            output[(y * width + x) * channels + c] = window[cnt/2];
        //}
    }
}

__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels, double* kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    int halfKernelSize = kernelSize / 2;

    if (x < width && y < height) {
        double sum = 0.0;
        double norm = 0.0;
        for (int ky = -halfKernelSize; ky <= halfKernelSize; ky++) {
            for (int kx = -halfKernelSize; kx <= halfKernelSize; kx++) {
                int ny = y + ky;
                int nx = x + kx;
                if (0 <= ny < height && 0 <= nx < width) {
                    double kernelValue = kernel[(ky + halfKernelSize) * kernelSize + (kx + halfKernelSize)];
                    sum += input[(ny * width + nx) * channels + c] * kernelValue;
                    norm += kernelValue;
                }
            }
        }
        output[(y * width + x) * channels + c] = sum / norm;
    }
}

void applyGaussianBlur(const cv::Mat& inputImage, cv::Mat& outputImage, double* kernel, int kernelSize) {
    int imageSize = inputImage.rows * inputImage.cols * inputImage.channels();
    unsigned char* d_input, * d_output;
    double* d_kernel;

    cudaMalloc(&d_input, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_output, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(double));

    cudaMemcpy(d_input, inputImage.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16, inputImage.channels());
    dim3 blocksPerGrid((inputImage.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputImage.rows + threadsPerBlock.y - 1) / threadsPerBlock.y, inputImage.channels());
    gaussianBlurKernel <<<blocksPerGrid, threadsPerBlock>>> (d_input, d_output, inputImage.cols, inputImage.rows, inputImage.channels(), d_kernel, kernelSize);

    cudaMemcpy(outputImage.data, d_output, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

void applyMedian(const cv::Mat& inputImage, cv::Mat& outputImage, unsigned char* kernel, int kernelSize) {
    int imageSize = inputImage.rows * inputImage.cols * inputImage.channels();
    unsigned char* d_input, * d_output;

    cudaMalloc(&d_input, imageSize * sizeof(unsigned char));
    cudaMalloc(&d_output, imageSize * sizeof(unsigned char));

    cudaMemcpy(d_input, inputImage.data, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16, inputImage.channels());
    dim3 blocksPerGrid((inputImage.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputImage.rows + threadsPerBlock.y - 1) / threadsPerBlock.y, inputImage.channels());
    MedianKernel <<<blocksPerGrid, threadsPerBlock >>> (d_input, d_output, inputImage.cols, inputImage.rows, inputImage.channels());

    cudaMemcpy(outputImage.data, d_output, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::Mat inputImage = cv::imread("mobile.png");
    if (inputImage.empty()) {
        std::cerr << "Error: Image cannot be loaded!" << std::endl;
        return -1;
    }

    cv::Mat outputImage1 = inputImage.clone();
    cv::Mat outputImage2 = inputImage.clone();
    cv::Mat outputImage3 = inputImage.clone();

    // Define Gaussian kernels
    double kernel1[] = { 1, 4, 6, 4, 1,
                       4, 16, 24, 16, 4,
                       6, 24, 36, 24, 6,
                       4, 16, 24, 16, 4,
                       1, 4, 6, 4, 1 };

    double kernel2[] = { 1, 2, 1,
                       2, 4, 2,
                       1, 2, 1 };

    int kernelSize1 = 5;
    int kernelSize2 = 3;
    int windowSize = 3;

    unsigned char* window;
    window = new unsigned char[windowSize];


    applyGaussianBlur(inputImage, outputImage1, kernel1, kernelSize1);
    applyGaussianBlur(inputImage, outputImage2, kernel2, kernelSize2);
    applyMedian(inputImage, outputImage3, window, windowSize);

    delete[] window;

    cv::imwrite("output_blur1.jpg", outputImage1);
    cv::imwrite("output_blur2.jpg", outputImage2);
    cv::imwrite("output_Med.jpg", outputImage3);

    return 0;
}
