#ifndef __CUDA_BLOCK_CUHPP__
#define __CUDA_BLOCK_CUHPP__
/*===========================================================================================
	- 必須要在 __global__ 函式裡 
	- 

============================================================================================*/
#define __cuda_parallel_1d__(size)															\
			int NThreads = gridDim.x*blockDim.x; 											\
			int NLoops; 																	\
			if(size < NThreads){															\
				NLoops = 1;																	\
			}else if((size%NThreads) == 0){													\
				NLoops = size/NThreads;														\
			}else{																			\
				NLoops = size/NThreads + 1;													\
			}																				\
																							\
			int idTheard = blockIdx.x*blockDim.x+threadIdx.x;								\
			for(int idLoop=0,idx=idTheard; 													\
				idLoop<NLoops;																\
				idLoop++,idx=(idTheard+idLoop*NThreads)										\
			) if(idx < size)																\
//=============================================================================================





#endif