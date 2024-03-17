// Retrieves the device properties
#include <iostream>

int main() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Device name " << prop.name << std::endl;
    std::cout << "Total Memory (GB):" << prop.totalGlobalMem / std::pow(1024, 3)
              << std::endl;
    std::cout << "Shared Memory per Block: (B) " << prop.sharedMemPerBlock
              << std::endl;
    std::cout << "registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Register per SM: " << prop.regsPerMultiprocessor << std::endl;
}
