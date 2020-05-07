
#ifndef __ASSIGN_CUHPP__
#define __ASSIGN_CUHPP__






// 平行化實作
__global__ void printPtr(unsigned char *ptr,int size){
	// 如果　總執行緒 大於向量長度
	int NThreads = gridDim.x*blockDim.x;
	int NLoop;
	if(size < NThreads){
		NLoop = 1;
	}else if((size%NThreads) == 0){
		NLoop = size/NThreads;
	}else{
		NLoop = size/NThreads + 1;
	}//endelse
	for(int i=0;i<NLoop;i++){  
		int id = blockIdx.x*blockDim.x+threadIdx.x;
		int pos = id + i*NThreads;
		if(pos < size){
			printf("ID : {%d x %d + %d = %d / %d}, ptr[%d / %d]:%d \n",blockIdx.x,blockDim.x,threadIdx.x,gridDim.x*blockDim.x,id,pos,size,(int)ptr[pos]);
		}//endif
	}//endfor
	
}




#endif