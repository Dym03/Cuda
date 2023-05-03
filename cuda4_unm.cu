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
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"

// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;

    // Get point from color picture
    // uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[ l_y * t_color_cuda_img.m_size.x + l_x ];
    uchar3 l_bgr = t_color_cuda_img.uchar3_point( l_x, l_y );

    // Store BW point to new image
    t_bw_cuda_img.m_p_uchar1[ l_y * t_bw_cuda_img.m_size.x + l_x ].x = l_bgr.x * 0.11 + l_bgr.y * 0.59 + l_bgr.z * 0.30;
}

void cu_run_grayscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_grayscale<<< l_blocks, l_threads >>>( t_color_cuda_img, t_bw_cuda_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}

__global__ void kernel_halfscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;

    // Get point from color picture
    uchar3 l_bgr = t_color_cuda_img.m_p_uchar3[ l_y * t_color_cuda_img.m_size.x + l_x ];

    // Store BW point to new image
    t_bw_cuda_img.m_p_uchar3[ l_y * t_bw_cuda_img.m_size.x + l_x ].x = l_bgr.x * 0.5;
    t_bw_cuda_img.m_p_uchar3[ l_y * t_bw_cuda_img.m_size.x + l_x ].y = l_bgr.y * 0.5;
    t_bw_cuda_img.m_p_uchar3[ l_y * t_bw_cuda_img.m_size.x + l_x ].z = l_bgr.z * 0.5;
}

void cu_run_halfscale( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_halfscale<<< l_blocks, l_threads >>>( t_color_cuda_img, t_bw_cuda_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}


__global__ void kernel_rotate( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;

    // Store BW point to new image
    t_bw_cuda_img.uchar3_point( l_y, l_x ) = t_color_cuda_img.uchar3_point( l_x, l_y );;
}

void cu_run_rotate( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_rotate<<< l_blocks, l_threads >>>( t_color_cuda_img, t_bw_cuda_img );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}

//https://stackoverflow.com/questions/9833316/cuda-image-rotation
__global__ void Rotate(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img , float deg)
{
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    if ( l_x >= t_color_cuda_img.m_size.x ) return;
    if ( l_y >= t_color_cuda_img.m_size.y ) return;

    int xc = t_color_cuda_img.m_size.x/2;
    int yc = t_color_cuda_img.m_size.y/2;
    deg = deg * M_PI / 180.0;
    int newx = ((float)l_x-xc)*cos(deg) - ((float)l_y-yc)*sin(deg) + xc;
    int newy = ((float)l_x-xc)*sin(deg) + ((float)l_y-yc)*cos(deg) + yc;
    if (newx >= 0 && newx < t_color_cuda_img.m_size.x && newy >= 0 && newy < t_color_cuda_img.m_size.y)
    {   
        t_bw_cuda_img.uchar3_point( l_x, l_y ) = t_color_cuda_img.uchar3_point( newx, newy );
    }else {
        t_bw_cuda_img.uchar3_point( l_x, l_y ) = make_uchar3(0,0,0);
    }
}

// __global__ void Rotate(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img, float deg) {
// 	int x = blockDim.x * blockIdx.x + threadIdx.x;
// 	int y = blockDim.y * blockIdx.y + threadIdx.y;
// 	if(x >= t_color_cuda_img.m_size.x || y >= t_color_cuda_img.m_size.y) {
// 		return;
// 	}


// 	int sx = t_color_cuda_img.m_size.x / 2;
// 	int sy = t_color_cuda_img.m_size.y / 2;

// 	float theta = deg * 3.14 / 180;
// 	int x2 = (x-sx) * cos(theta) - (y - sy) * sin(theta) + sx;
// 	int y2 = (x-sx) * sin(theta) + (y - sy) * cos(theta) + sy;

// 	if(x2 >= 0 && x2 < t_color_cuda_img.m_size.x && y2 >=0 && y2 < t_color_cuda_img.m_size.y) {
// 		t_bw_cuda_img.uchar3_point(x, y) = t_color_cuda_img.uchar3_point(x2, y2);
// 	} else {
// 		t_bw_cuda_img.uchar3_point(x, y) = (uchar3) {0, 0, 0};
// 	}
// }

// __global__ void rotateImage_Kernel(CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img, const float angle, const float scale)
// {
//     // compute thread dimension
//     const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if ( y >= t_color_cuda_img.m_size.y ) return;
//     if ( x >= t_color_cuda_img.m_size.x ) return;

//     //// compute target address
//     const unsigned int idx = x + y * t_color_cuda_img.m_size.x;

//     const int xA = (x - t_color_cuda_img.m_size.x/2 );
//     const int yA = (y - t_color_cuda_img.m_size.y/2 );

//     const int xR = (int)floor(1.0f/scale * (xA * cos(angle) - yA * sin(angle)));
//     const int yR = (int)floor(1.0f/scale * (xA * sin(angle) + yA * cos(angle)));

//     float src_x = xR + t_color_cuda_img.m_size.x/2;
//     float src_y = yR + t_color_cuda_img.m_size.y/2;



//      if ( src_x >= 0.0f && src_x < t_color_cuda_img.m_size.x && src_y >= 0.0f && src_y < t_color_cuda_img.m_size.y) {
//         // BI - LINEAR INTERPOLATION
//         float src_x0 = (float)(int)(src_x);
//         float src_x1 = (src_x0+1);
//         float src_y0 = (float)(int)(src_y);
//         float src_y1 = (src_y0+1);

//         float sx = (src_x-src_x0);
//         float sy = (src_y-src_y0);


//         int idx_src00 = min(max(0.0f,src_x0   + src_y0 * t_color_cuda_img.m_size.x),t_color_cuda_img.m_size.x*t_color_cuda_img.m_size.y-1.0f);
//         int idx_src10 = min(max(0.0f,src_x1   + src_y0 * t_color_cuda_img.m_size.x),t_color_cuda_img.m_size.x*t_color_cuda_img.m_size.y-1.0f);
//         int idx_src01 = min(max(0.0f,src_x0   + src_y1 * t_color_cuda_img.m_size.x),t_color_cuda_img.m_size.x*t_color_cuda_img.m_size.y-1.0f);
//         int idx_src11 = min(max(0.0f,src_x1   + src_y1 * t_color_cuda_img.m_size.x),t_color_cuda_img.m_size.x*t_color_cuda_img.m_size.y-1.0f);

//         trg[idx].y = 0.0f;

//         trg[idx].x  = (1.0f-sx)*(1.0f-sy)*src[idx_src00].x;
//         trg[idx].x += (     sx)*(1.0f-sy)*src[idx_src10].x;
//         trg[idx].x += (1.0f-sx)*(     sy)*src[idx_src01].x;
//         trg[idx].x += (     sx)*(     sy)*src[idx_src11].x;
//     } else {
//         trg[idx].x = 0.0f;
//         trg[idx].y = 0.0f;
//      }

//     DEVICE_METHODE_LAST_COMMAND;

// }



void cu_run_rotate_2( CudaImg t_color_cuda_img, CudaImg t_bw_cuda_img , float deg)
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks( ( t_color_cuda_img.m_size.x + l_block_size - 1 ) / l_block_size, ( t_color_cuda_img.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    Rotate<<< l_blocks, l_threads >>>( t_color_cuda_img, t_bw_cuda_img , deg);

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}

// Demo kernel to create picture with alpha channel gradient
__global__ void kernel_insertimage( CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position )
{
    // X,Y coordinates and check image dimensions
    int l_y = blockDim.y * blockIdx.y + threadIdx.y;
    int l_x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( l_y >= t_small_cuda_pic.m_size.y ) return;
    if ( l_x >= t_small_cuda_pic.m_size.x ) return;
    int l_by = l_y + t_position.y;
    int l_bx = l_x + t_position.x;
    if ( l_by >= t_big_cuda_pic.m_size.y || l_by < 0 ) return;
    if ( l_bx >= t_big_cuda_pic.m_size.x || l_bx < 0 ) return;

    // Get point from small image
    uchar4 l_fg_bgra = t_small_cuda_pic.m_p_uchar4[ l_y * t_small_cuda_pic.m_size.x + l_x ];
    uchar3 l_bg_bgr = t_big_cuda_pic.m_p_uchar3[ l_by * t_big_cuda_pic.m_size.x + l_bx ];
    uchar3 l_bgr = { 0, 0, 0 };

    // compose point from small and big image according alpha channel
    l_bgr.x = l_fg_bgra.x * l_fg_bgra.w / 255 + l_bg_bgr.x * ( 255 - l_fg_bgra.w ) / 255;
    l_bgr.y = l_fg_bgra.y * l_fg_bgra.w / 255 + l_bg_bgr.y * ( 255 - l_fg_bgra.w ) / 255;
    l_bgr.z = l_fg_bgra.z * l_fg_bgra.w / 255 + l_bg_bgr.z * ( 255 - l_fg_bgra.w ) / 255;

    // Store point into image
    t_big_cuda_pic.m_p_uchar3[ l_by * t_big_cuda_pic.m_size.x + l_bx ] = l_bgr;
}

void cu_insertimage( CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position )
{
    cudaError_t l_cerr;

    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks( ( t_small_cuda_pic.m_size.x + l_block_size - 1 ) / l_block_size,
                   ( t_small_cuda_pic.m_size.y + l_block_size - 1 ) / l_block_size );
    dim3 l_threads( l_block_size, l_block_size );
    kernel_insertimage<<< l_blocks, l_threads >>>( t_big_cuda_pic, t_small_cuda_pic, t_position );

    if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

    cudaDeviceSynchronize();
}


// __global__ void resize_kernel( CudaImg in_pic, CudaImg out_pic)
// {
//     int i = blockDim.y * blockIdx.y + threadIdx.y;
//     int j = blockDim.x * blockIdx.x + threadIdx.x;

//     int channel = 3;

//     if( i < out_pic. && j < widthOut )
//     {
//         int iIn = i * heightIn / heightOut;
//         int jIn = j * widthIn / widthOut;
//         for(int c = 0; c < channel; c++)
//             pOut[ (i*widthOut + j)*channel + c ] = pIn[ (iIn*widthIn + jIn)*channel + c ];
//     }
// }

// void cu_resize( CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic )
// {
//     cudaError_t l_cerr;

//     // Grid creation, size of grid must be equal or greater than images
//     int l_block_size = 32;
//     dim3 l_blocks( ( t_small_cuda_pic.m_size.x + l_block_size - 1 ) / l_block_size,
//                    ( t_small_cuda_pic.m_size.y + l_block_size - 1 ) / l_block_size );
//     dim3 l_threads( l_block_size, l_block_size );
//     kernel_insertimage<<< l_blocks, l_threads >>>( t_big_cuda_pic, t_small_cuda_pic, t_position );

//     if ( ( l_cerr = cudaGetLastError() ) != cudaSuccess )
//         printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( l_cerr ) );

//     cudaDeviceSynchronize();
// }


__global__ void kernel(CudaImg in_pic, CudaImg out_pic) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= out_pic.m_size.x || y >= out_pic.m_size.y) {
		return;
	}
    
    uchar3 px1 = in_pic.uchar3_point(x*2, y*2);
    uchar3 px2 = in_pic.uchar3_point(x*2 + 1, y*2);
    

    out_pic.m_p_uchar3[y * out_pic.m_size.x + x].x = (px1.x + px2.x) / 2;
    out_pic.m_p_uchar3[y * out_pic.m_size.x + x].y = (px1.y + px2.y) / 2;
    out_pic.m_p_uchar3[y * out_pic.m_size.x + x].z = (px1.z + px2.z) / 2;
}

void cuda_resize(CudaImg in_pic, CudaImg out_pic) {

	int count = 10;
	dim3 blocks((out_pic.m_size.x + count)/ count, (out_pic.m_size.y + count) / count);
	dim3 threads(count, count);
	kernel<<<blocks, threads>>>(in_pic, out_pic);

}


__global__ void kernel_alpha(CudaImg in_pic, CudaImg out_pic) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= out_pic.m_size.x || y >= out_pic.m_size.y) {
		return;
	}
    
    uchar4 px1 = in_pic.uchar4_point(x*2, y*2);
    uchar4 px2 = in_pic.uchar4_point(x*2 + 1, y*2);
    
    out_pic.m_p_uchar4[y * out_pic.m_size.x + x].x = (px1.x + px2.x) / 2;
    out_pic.m_p_uchar4[y * out_pic.m_size.x + x].y = (px1.y + px2.y) / 2;
    out_pic.m_p_uchar4[y * out_pic.m_size.x + x].z = (px1.z + px2.z) / 2;
    out_pic.m_p_uchar4[y * out_pic.m_size.x + x].w = (px1.w + px2.w) / 2;
}

void cuda_resize_alpha(CudaImg in_pic, CudaImg out_pic) {
    if(in_pic.m_channels != 4 || out_pic.m_channels != 4) {
        printf("Error: cuda_resize_alpha only support 4 channels\n");
        return;
    }

	int count = 10;
	dim3 blocks((out_pic.m_size.x + count)/ count, (out_pic.m_size.y + count) / count);
	dim3 threads(count, count);
	kernel_alpha<<<blocks, threads>>>(in_pic, out_pic);
    cudaDeviceSynchronize();
}

__global__ void rotate_first_last(CudaImg in_pic, CudaImg out_pic) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= in_pic.m_size.x || y >= in_pic.m_size.y) {
		return;
	}
    
    if((x <= (int) (in_pic.m_size.x / 3) && y <= (int) (in_pic.m_size.y / 3) ) || (x >= (int)(2 * in_pic.m_size.x / 3) && y >= (int)(2 * in_pic.m_size.y / 3))) {
        int out_x = y;
        int out_y = x;
        // if (x <= (int) (in_pic.m_size.x / 3)) {
        //     out_x = in_pic.m_size.y - out_x - 1;
        // } else {
        //     out_y = in_pic.m_size.x - out_y - 1;
        // }
        
        out_pic.uchar3_point(x, y) = in_pic.uchar3_point(out_x, out_y);
    } else {
        out_pic.uchar3_point(x, y) = in_pic.uchar3_point(x, y);
    }
}


void cuda_rotate_first_last(CudaImg in_pic, CudaImg out_pic) {

	int count = 10;
	dim3 blocks((out_pic.m_size.x + count)/ count, (out_pic.m_size.y + count) / count);
	dim3 threads(count, count);
	rotate_first_last<<<blocks, threads>>>(in_pic, out_pic);
    cudaDeviceSynchronize();
}