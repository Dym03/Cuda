#include "cuda_img.h"

__host__ __device__ CudaImg::CudaImg(cv::Mat& i_mat) {
    this->m_size.x = i_mat.size().width;
    this->m_size.y = i_mat.size().height;
    if (i_mat.channels() == 1) {
        this->m_p_uchar1 = (uchar1*)i_mat.data;
        this->m_channels = 1;
    } else if (i_mat.channels() == 3) {
        this->m_p_uchar3 = (uchar3*)i_mat.data;
        this->m_channels = 3;
    } else if (i_mat.channels() == 4) {
        this->m_p_uchar4 = (uchar4*)i_mat.data;
        this->m_channels = 4;
    }
}

CudaImg::~CudaImg() {
}
