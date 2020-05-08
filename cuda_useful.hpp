
#ifndef __CUDA_USEFUL_CUHPP__
#define __CUDA_USEFUL_CUHPP__

#include "cuda_block.hpp"

namespace cuda_useful{


	/*============================================================
	// CUDA_PARALLEL_1D{　掃全部　, 列印此工作是給哪個工人負責?? }　
	==============================================================*/
	__global__ void printPixel(unsigned char *ptr,int size){
		__cuda_parallel_1d__(size){
			printf("ID : {%d x %d + %d = %d / %d}, ptr[%d / %d]:%d \n",blockIdx.x,blockDim.x,threadIdx.x,NThreads,idLoop,idx,size,(int)ptr[idx]);
		}
	}
	
	/*===============================================================
	// CUDA_PARALLEL_1D{ 掃全部 ,　設定 BGR  }　
	=================================================================*/

	__global__ void setConstantBGR(unsigned char *ptr,int size,int b,int g,int r){
		__cuda_parallel_1d__(size){
			int v = idx%3;
			if(v==0){
				ptr[idx] = (unsigned char)b;
			}else if(v==1){
				ptr[idx] = (unsigned char)g;
			}else{
				ptr[idx] = (unsigned char)r;
			}//end_else
		}
	}

	/*==========================================================================
	CUDA_PARALLEL_1D{ 　掃更新 , 給定座標　(x,y) ---> 設定 BGR }
	===========================================================================*/
	__global__ void setBGRs(
		unsigned char *ptr,
		int*x,
		int*y,
		int w,
		unsigned char *b,
		unsigned char *g,
		unsigned char *r,
		int size){
		__cuda_parallel_1d__(size){
				int bgrIdx = 3*(y[idx]*w + x[idx]);
				ptr[bgrIdx] = b[idx];
				ptr[bgrIdx+1] = g[idx];
				ptr[bgrIdx+2] = r[idx];
		}
	}//end_setBGR



};



#endif