/**
 * @file main.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2026-05-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

 #include <gtest/gtest.h>

class CudaEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        // cuda warmup to avoid context initialization costs
        void* tmp;
        cudaMalloc(&tmp, 1);
        cudaFree(tmp);
        cudaDeviceSynchronize();
    }
};

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment());
    return RUN_ALL_TESTS();
}