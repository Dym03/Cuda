#include "cuda_img.h"

__host__ __device__ uchar3 &CudaImg::uchar3_point(int i_x, int i_y) {
    // if(this->m_p_uchar3 == nullptr)
    //     printf("m_p_uchar3 is null\n");
    return this->m_p_uchar3[i_y * this->m_size.x + i_x];
}

__host__ __device__ uchar1 &CudaImg::uchar1_point(int i_x, int i_y) {
    if(this->m_p_uchar1 == nullptr)
        printf("m_p_uchar1 is null\n");
    return this->m_p_uchar1[i_y * this->m_size.x + i_x];
}

__host__ __device__ uchar4 &CudaImg::uchar4_point(int i_x, int i_y) {
    if(this->m_p_uchar4 == nullptr)
        printf("m_p_uchar4 is null\n");
    return this->m_p_uchar4[i_y * this->m_size.x + i_x];
}