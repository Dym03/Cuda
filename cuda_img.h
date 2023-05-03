// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once
#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"


// Structure definition for exchanging data between Host and Device
struct CudaImg
{
  public:
  uint3 m_size;             // size of picture
  uint m_channels;          // number of channels
  union {
      void   *m_p_void;     // data of picture
      uchar1 *m_p_uchar1;   // data of picture
      uchar3 *m_p_uchar3;   // data of picture
      uchar4 *m_p_uchar4;   // data of picture
  };

    // Constructor
    CudaImg( cv::Mat &i_mat );

    // Destructor
    ~CudaImg();

    // Access to pixel
    __host__ __device__ uchar3& uchar3_point( int i_x, int i_y );
    // Access to pixel
    __host__ __device__ uchar1& uchar1_point( int i_x, int i_y );
    // Access to pixel
    __host__ __device__ uchar4& uchar4_point( int i_x, int i_y );
};
