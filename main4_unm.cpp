// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <iostream>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototype from .cu file
void cu_run_grayscale( CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img );
void cu_run_halfscale( CudaImg t_bgr_cuda_img, CudaImg t_bw_cuda_img );
void cu_run_rotate( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img );
void cu_run_rotate_2( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img , float deg);
void cu_insertimage( CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position );
void cuda_resize(CudaImg in_pic, CudaImg out_pic);
void cuda_resize_alpha(CudaImg in_pic, CudaImg out_pic);
void cuda_rotate_first_last(CudaImg in_pic, CudaImg out_pic);

int main( int t_numarg, char **t_arg )
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator( &allocator );

    // if ( t_numarg < 2 )
    // {
    //     printf( "Enter picture filename!\n" );
    //     return 1;
    // }

    // // Load image
    // cv::Mat l_bgr_cv_img = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR ); // CV_LOAD_IMAGE_COLOR );

    // if ( !l_bgr_cv_img.data )
    // {
    //     printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
    //     return 1;
    // }

    // create empty BW image
    cv::Mat kamen_cv = cv::imread("kamen.jpg", cv::IMREAD_COLOR);
    // cv::Mat smaller = cv::imread("kamen.jpg", cv::IMREAD_COLOR);
    cv::Mat smaller(kamen_cv.size(), CV_8UC3);
    cv::Mat baseball_cv = cv::imread("ball.png", cv::IMREAD_UNCHANGED);
    printf("%d %d", kamen_cv.channels(), kamen_cv.channels());
    CudaImg kamen(kamen_cv);
    CudaImg baseball(baseball_cv);
    printf("%d", smaller.channels());
    CudaImg smaller_cuda(smaller);
    int2 pos = {300, 50};
    cuda_rotate_first_last(kamen, smaller_cuda);
    cv::imshow("smaller", smaller);
    // printf("%d", smaller.channels());
    // cu_run_rotate_2(baseball, smaller_cuda, 45);
    // cu_insertimage(kamen, baseball, pos);
    // cv::imshow("kamen", kamen_cv);
    // printf("Kamen %d", kamen.m_channels);
    // cuda_resize(kamen, smaller_cuda);
    // cv::imshow("smaller", smaller);
    // cv::imshow("smaller", smaller);
    // cv::imshow("kamen", kamen_cv);
    // cv::Mat l_bw_cv_img(l_bgr_cv_img.size(), CV_8UC3);
    // int tmp = 0;
    // tmp = l_bgr_cv_img.size().width;
    // cv::Mat l_bw_cv_img(tmp, l_bgr_cv_img.size().height, CV_8UC3 );
    // std::cout << l_bw_cv_img.size() << std::endl;

    // data for CUDA
    // CudaImg l_bgr_cuda_img, l_bw_cuda_img;
    // l_bgr_cuda_img.m_size.x = l_bw_cuda_img.m_size.x = l_bgr_cv_img.size().width;
    // l_bgr_cuda_img.m_size.y = l_bw_cuda_img.m_size.y = l_bgr_cv_img.size().height;
    // l_bgr_cuda_img.m_p_uchar3 = ( uchar3 * ) l_bgr_cv_img.data;
    // l_bw_cuda_img.m_p_uchar1 = ( uchar1 * ) l_bw_cv_img.data;

    // // Function calling from .cu file
    // cu_run_grayscale( l_bgr_cuda_img, l_bw_cuda_img );
    // CudaImg original(l_bgr_cv_img);
    // CudaImg svetly(l_bw_cv_img);
    // uchar3 original_data = original.uchar3_point(0,0);
    // cu_run_halfscale(original, svetly);
    // cu_run_rotate_2(original, svetly, 180);
    // cu_run_rotate(original, svetly);

    // Show the Color and BW image
    // cv::imshow( "Color", l_bgr_cv_img );
    // cv::imshow( "GrayScale", l_bw_cv_img );
    // cv::imshow("svelty", svetly);
    cv::waitKey( 0 );
}

