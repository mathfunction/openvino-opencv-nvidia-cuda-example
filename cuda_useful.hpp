
#ifndef __CUDA_USEFUL_CUHPP__
#define __CUDA_USEFUL_CUHPP__



namespace cuda_useful{


	__device__ int checkLoop1d(int NThreads,int size){
		if(size < NThreads){
			return 1;
		}else if((size%NThreads) == 0){
			return size/NThreads;
		}else{
			return size/NThreads + 1;
		}//endelse
	}


	/*============================================================
	// CUDA_PARALLEL_1D{　掃全部　, 列印此工作是給哪個工人負責?? }　
	==============================================================*/
	__global__ void printPixel(unsigned char *ptr,int size){
		int NThreads = gridDim.x*blockDim.x;
		int NLoop = checkLoop1d(NThreads,size);
		for(int i=0;i<NLoop;i++){  
			int id = blockIdx.x*blockDim.x+threadIdx.x;
			int pos = id + i*NThreads;
			if(pos < size){
				// =====================================================================================================================================================
				printf("ID : {%d x %d + %d = %d / %d}, ptr[%d / %d]:%d \n",blockIdx.x,blockDim.x,threadIdx.x,gridDim.x*blockDim.x,id,pos,size,(int)ptr[pos]);
				//=======================================================================================================================================================
			}//endif
		}//endfor
		
	}

	/*===============================================================
	// CUDA_PARALLEL_1D{ 掃全部 ,　設定 BGR  }　
	=================================================================*/
	__global__ void setConstantBGR(unsigned char *ptr,int size,int b,int g,int r){
		int NThreads = gridDim.x*blockDim.x;
		int NLoop = checkLoop1d(NThreads,size);
		for(int i=0;i<NLoop;i++){  
			int id = blockIdx.x*blockDim.x+threadIdx.x;
			int pos = id + i*NThreads;
			// 平行化　BGR 
			if(pos < size){
				int v = pos%3;
				if(v==0){
					ptr[pos] = (unsigned char)b;
				}else if(v==1){
					ptr[pos] = (unsigned char)g;
				}else{
					ptr[pos] = (unsigned char)r;
				}//end_else
			}//endif
		}//endfor

	}

	/*==========================================================================
	CUDA_PARALLEL_1D{ 　掃更新 , 給定座標　(x,y) ---> 設定 BGR }
	===========================================================================*/
	__global__ void setBGRs(
		unsigned char *ptr,
		int*x,
		int*y,
		int w,
		int*b,
		int*g,
		int*r,
		int size)
	{
		int NThreads = gridDim.x*blockDim.x;
		int NLoop = checkLoop1d(NThreads,size);
		for(int i=0;i<NLoop;i++){  
			int id = blockIdx.x*blockDim.x+threadIdx.x;
			int pos = id + i*NThreads;
			if(pos < size){
				int bgrIdx = 3*(y[pos]*w + x[pos]);
				ptr[bgrIdx] = b[pos];
				ptr[bgrIdx] = g[pos];
				ptr[bgrIdx] = r[pos];
			}//endif
		}//endfor	
	}//end_setBGR





};



#endif