#pragma warning(disable:4996)
#define NOMINMAX

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLUR_RADIUS 2
#define BLOCK_SIZE  16

// Shared memory tile size = block + halo border on all sides
#define TILE_SIZE (BLOCK_SIZE + 2 * BLUR_RADIUS)  // 16 + 4 = 20

int TEST_WIDTHS[]  = {512,  1024, 1920};
int TEST_HEIGHTS[] = {512,  1024, 1080};
#define NUM_TESTS 3

// ── CUDA error checking ───────────────────────────────────
#define CUDA_CHECK(call) {                                          \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        printf("CUDA ERROR %d at line %d [%s]: %s\n",              \
               (int)err, __LINE__, #call,                           \
               cudaGetErrorString(err));                            \
        exit(1);                                                    \
    }                                                               \
}

// ── Bilinear resize ───────────────────────────────────────
void resizeImage(unsigned char* src, int srcW, int srcH,
                 unsigned char* dst, int dstW, int dstH)
{
    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            float fx = (float)x * (float)(srcW - 1) / (float)(dstW - 1);
            float fy = (float)y * (float)(srcH - 1) / (float)(dstH - 1);

            int x0 = (int)fx;
            int y0 = (int)fy;
            int x1 = x0 + 1; if (x1 >= srcW) x1 = srcW - 1;
            int y1 = y0 + 1; if (y1 >= srcH) y1 = srcH - 1;

            float wx = fx - x0;
            float wy = fy - y0;

            float val =
                (1-wx)*(1-wy) * src[y0*srcW + x0] +
                   wx *(1-wy) * src[y0*srcW + x1] +
                (1-wx)*   wy  * src[y1*srcW + x0] +
                   wx *   wy  * src[y1*srcW + x1];

            dst[y * dstW + x] = (unsigned char)(val + 0.5f);
        }
    }
}

// ── GPU Kernel (Shared Memory) ────────────────────────────
__global__ void boxBlurKernel(
    unsigned char* input, unsigned char* output,
    int width, int height)
{
    // ── Declare shared memory tile ───────────────────────
    // Each block loads a TILE_SIZE x TILE_SIZE region into
    // fast on-chip shared memory, including the halo border
    __shared__ unsigned char tile[TILE_SIZE][TILE_SIZE];

    // ── Thread and pixel coordinates ─────────────────────
    int tx = threadIdx.x;  // thread position inside block (0-15)
    int ty = threadIdx.y;

    // Global pixel this thread is responsible for outputting
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    // ── Step 1: Load tile into shared memory ─────────────
    // Each thread loads its own pixel PLUS some halo pixels.
    // The tile is TILE_SIZE x TILE_SIZE (20x20).
    // We use a loop so every cell of the tile gets loaded
    // even if the block has fewer threads than tile cells.

    for (int row = ty; row < TILE_SIZE; row += BLOCK_SIZE) {
        for (int col = tx; col < TILE_SIZE; col += BLOCK_SIZE) {

            // Global source coordinates for this tile cell
            // Offset by BLUR_RADIUS so halo starts at block edge
            int srcX = blockIdx.x * BLOCK_SIZE + col - BLUR_RADIUS;
            int srcY = blockIdx.y * BLOCK_SIZE + row - BLUR_RADIUS;

            // Clamp to image boundary (same clamp-to-edge as before)
            if (srcX < 0)      srcX = 0;
            if (srcX >= width)  srcX = width  - 1;
            if (srcY < 0)      srcY = 0;
            if (srcY >= height) srcY = height - 1;

            // Load from GLOBAL memory into SHARED memory
            tile[row][col] = input[srcY * width + srcX];
        }
    }

    // ── Step 2: Synchronise all threads in the block ─────
    // CRITICAL: Every thread must finish loading before any
    // thread starts reading. Without this, some threads
    // might read uninitialised shared memory values.
    __syncthreads();

    // ── Step 3: Compute blur from shared memory ──────────
    // Threads that fall outside the image do not write output
    // but still participated in loading the tile above
    if (x >= width || y >= height) return;

    int sum = 0, count = 0;
    for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++) {
        for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++) {

            // Read from SHARED memory instead of global memory
            // tx + BLUR_RADIUS = this thread's position in the tile
            // + kx/ky = neighbour offset within the tile
            sum += tile[ty + BLUR_RADIUS + ky][tx + BLUR_RADIUS + kx];
            count++;
        }
    }

    output[y * width + x] = (unsigned char)(sum / count);
}

// ── CPU Reference ─────────────────────────────────────────
void boxBlurCPU(
    unsigned char* input, unsigned char* output,
    int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0, count = 0;
            for (int ky = -BLUR_RADIUS; ky <= BLUR_RADIUS; ky++) {
                for (int kx = -BLUR_RADIUS; kx <= BLUR_RADIUS; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx < 0) nx = 0;
                    if (nx >= width)  nx = width  - 1;
                    if (ny < 0) ny = 0;
                    if (ny >= height) ny = height - 1;
                    sum += input[ny * width + nx];
                    count++;
                }
            }
            output[y * width + x] = (unsigned char)(sum / count);
        }
    }
}

// ── PSNR ──────────────────────────────────────────────────
double computePSNR(unsigned char* original, unsigned char* blurred, int n)
{
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)original[i] - (double)blurred[i];
        mse += d * d;
    }
    mse /= n;
    if (mse == 0.0) return 100.0;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

// ── Save greyscale as JPEG ────────────────────────────────
void saveGreyscaleJPG(const char* fname,
                      unsigned char* grey, int w, int h)
{
    unsigned char* rgb = (unsigned char*)malloc(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        rgb[i*3+0] = grey[i];
        rgb[i*3+1] = grey[i];
        rgb[i*3+2] = grey[i];
    }
    int result = stbi_write_jpg(fname, w, h, 3, rgb, 95);
    if (!result) printf("  WARNING: Failed to write %s\n", fname);
    free(rgb);
}

// ── Benchmark ─────────────────────────────────────────────
void runBenchmark(unsigned char* src, int srcW, int srcH,
                  int dstW, int dstH)
{
    int imgSize = dstW * dstH;

    unsigned char* resized   = (unsigned char*)malloc(imgSize);
    unsigned char* h_out_cpu = (unsigned char*)malloc(imgSize);
    unsigned char* h_out_gpu = (unsigned char*)malloc(imgSize);

    if (!resized || !h_out_cpu || !h_out_gpu) {
        printf("ERROR: malloc failed for %dx%d\n", dstW, dstH);
        return;
    }

    resizeImage(src, srcW, srcH, resized, dstW, dstH);

    // ── CPU timing ──────────────────────────────────────
    clock_t t0 = clock();
    boxBlurCPU(resized, h_out_cpu, dstW, dstH);
    clock_t t1 = clock();
    double cpuMs = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    // ── GPU setup ────────────────────────────────────────
    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  imgSize));
    CUDA_CHECK(cudaMalloc(&d_out, imgSize));
    CUDA_CHECK(cudaMemcpy(d_in, resized, imgSize, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((dstW + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (dstH + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ── GPU timing ───────────────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    boxBlurKernel<<<grid, block>>>(d_in, d_out, dstW, dstH);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, imgSize,
                          cudaMemcpyDeviceToHost));

    // ── Metrics ──────────────────────────────────────────
    double psnr    = computePSNR(resized, h_out_gpu, imgSize);
    double speedup = cpuMs / (double)gpuMs;

    printf("| %4dx%-4d | %8.2f ms | %7.3f ms | %7.2fx | %6.2f dB |\n",
           dstW, dstH, cpuMs, gpuMs, speedup, psnr);

    // ── Save outputs ─────────────────────────────────────
    char fname[64];
    sprintf(fname, "output_%dx%d.jpg", dstW, dstH);
    saveGreyscaleJPG(fname, h_out_gpu, dstW, dstH);

    // ── Cleanup ──────────────────────────────────────────
    free(resized); free(h_out_cpu); free(h_out_gpu);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ── Main ──────────────────────────────────────────────────
int main()
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s\n", prop.name);
    printf("VRAM: %.0f MB\n\n", prop.totalGlobalMem / 1048576.0f);

    int w, h, ch;
    unsigned char* img = stbi_load("input.jpg", &w, &h, &ch, 1);
    if (!img) {
        printf("ERROR: Cannot load input.jpg\n");
        return 1;
    }
    printf("Loaded: %dx%d (greyscale)\n\n", w, h);
    printf("Blur Kernel: %dx%d  |  Block: %dx%d  |  Tile: %dx%d (with halo)\n\n",
           2*BLUR_RADIUS+1, 2*BLUR_RADIUS+1,
           BLOCK_SIZE, BLOCK_SIZE,
           TILE_SIZE, TILE_SIZE);

    printf("+------------+-------------+------------+----------+-----------+\n");
    printf("| Resolution |   CPU Time  |  GPU Time  | Speedup  |   PSNR    |\n");
    printf("+------------+-------------+------------+----------+-----------+\n");

    for (int i = 0; i < NUM_TESTS; i++)
        runBenchmark(img, w, h, TEST_WIDTHS[i], TEST_HEIGHTS[i]);

    printf("+------------+-------------+------------+----------+-----------+\n");
    printf("\nDone. Output images saved.\n");

    stbi_image_free(img);
    return 0;
}