%%cuda
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

using namespace std;


// ============================================================
// CUDA Error Check
// ============================================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        cerr << "CUDA Error: "                                \
             << cudaGetErrorString(err)                       \
             << " at line " << __LINE__ << endl;              \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)


// ============================================================
// RGB -> Grayscale Kernel
// ============================================================

__global__ void colorToGrayscaleConversion(
    unsigned char* Pout,
    const unsigned char* Pin,
    int width,
    int height)
{
    int col =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int row =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (row < height && col < width) {

        // 灰階圖片的位置
        int grayOffset =
            row * width + col;

        // RGB 圖片的位置
        int rgbOffset =
            grayOffset * 3;

        unsigned char r =
            Pin[rgbOffset];

        unsigned char g =
            Pin[rgbOffset + 1];

        unsigned char b =
            Pin[rgbOffset + 2];

        // RGB -> Grayscale
        Pout[grayOffset] =
            static_cast<unsigned char>(
                0.21f * r +
                0.72f * g +
                0.07f * b
            );
    }
}


// ============================================================
// PPM Reader
// P6 格式
// ============================================================

bool readPPM(
    const string& filename,
    vector<unsigned char>& image,
    int& width,
    int& height)
{
    ifstream file(filename, ios::binary);

    if (!file) {
        cerr << "Cannot open file: "
             << filename << endl;
        return false;
    }

    string format;

    file >> format;

    if (format != "P6") {
        cerr << "Only P6 PPM format supported."
             << endl;
        return false;
    }

    // 跳過註解
    file >> ws;

    while (file.peek() == '#') {
        string line;
        getline(file, line);
        file >> ws;
    }

    int maxValue;

    file >> width;
    file >> height;
    file >> maxValue;

    file.get();

    if (maxValue != 255) {
        cerr << "Only max value 255 supported."
             << endl;
        return false;
    }

    size_t imageSize =
        width * height * 3;

    image.resize(imageSize);

    file.read(
        reinterpret_cast<char*>(image.data()),
        imageSize
    );

    return true;
}


// ============================================================
// PGM Writer
// P5 灰階格式
// ============================================================

bool writePGM(
    const string& filename,
    const vector<unsigned char>& image,
    int width,
    int height)
{
    ofstream file(filename, ios::binary);

    if (!file) {
        cerr << "Cannot create file: "
             << filename << endl;
        return false;
    }

    file << "P5\n";
    file << width << " "
         << height << "\n";
    file << "255\n";

    file.write(
        reinterpret_cast<const char*>(image.data()),
        width * height
    );

    return true;
}


// ============================================================
// main
// ============================================================

int main()
{
    string inputFile =
        "input.ppm";

    string outputFile =
        "output.ppm";


    // --------------------------------------------------------
    // 1. Host 讀取圖片
    // --------------------------------------------------------

    int width;
    int height;

    vector<unsigned char> h_rgb;

    if (!readPPM(
            inputFile,
            h_rgb,
            width,
            height))
    {
        return 1;
    }

    cout << "Image size: "
         << width
         << " x "
         << height
         << endl;


    // --------------------------------------------------------
    // 2. 計算記憶體大小
    // --------------------------------------------------------

    size_t rgbSize =
        width *
        height *
        3 *
        sizeof(unsigned char);

    size_t graySize =
        width *
        height *
        sizeof(unsigned char);


    vector<unsigned char> h_gray(
        width * height
    );


    // --------------------------------------------------------
    // 3. Device Memory
    // --------------------------------------------------------

    unsigned char* d_rgb;
    unsigned char* d_gray;

    CUDA_CHECK(
        cudaMalloc(
            &d_rgb,
            rgbSize
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_gray,
            graySize
        )
    );


    // --------------------------------------------------------
    // 4. Host -> Device
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_rgb,
            h_rgb.data(),
            rgbSize,
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // 5. 設定 Block / Grid
    // --------------------------------------------------------

    dim3 blockDim(
        16,
        16
    );

    dim3 gridDim(
        (width +
         blockDim.x - 1)
         / blockDim.x,

        (height +
         blockDim.y - 1)
         / blockDim.y
    );


    cout << "Block = "
         << blockDim.x
         << " x "
         << blockDim.y
         << endl;

    cout << "Grid = "
         << gridDim.x
         << " x "
         << gridDim.y
         << endl;


    // --------------------------------------------------------
    // 6. Launch Kernel
    // --------------------------------------------------------

    colorToGrayscaleConversion
        <<<gridDim, blockDim>>>(
            d_gray,
            d_rgb,
            width,
            height
        );


    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // --------------------------------------------------------
    // 7. Device -> Host
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_gray.data(),
            d_gray,
            graySize,
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // 8. 儲存灰階圖片
    // --------------------------------------------------------

    if (!writePGM(
            outputFile,
            h_gray,
            width,
            height))
    {
        cudaFree(d_rgb);
        cudaFree(d_gray);

        return 1;
    }


    cout << "Grayscale image saved to: "
         << outputFile
         << endl;


    // --------------------------------------------------------
    // 9. Free Device Memory
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaFree(d_rgb)
    );

    CUDA_CHECK(
        cudaFree(d_gray)
    );


    return 0;
}