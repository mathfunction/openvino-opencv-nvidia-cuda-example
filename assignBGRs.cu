/*==========================================================================================
// 這是關於把一個影片輸入，利用 GPU 運算 ，把彩色影片添加隨機對角線
"C:\\Program Files (x86)\\IntelSWTools\\openvino\\bin\\setupvars.bat"
// opencv == H x W x C (BGR)

=============================================================================================*/
#include <chrono>
#include <iostream>
#include <cstdlib>
// nvidia cuda library  
#include "npp.h"
#include "cublas.h"
#include "cusparse.h"
#include <thrust/device_vector.h>

// intel-openvino-opencv
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
// DIYCUDA
#include "cuda_useful.hpp"


using namespace std;








__global__ void setGPUChannels(
		int *dx,
		int *dy,
		unsigned char *dB,
		unsigned char *dG,
		unsigned char *dR,
		int size,
		int b,
		int g,
		int r){
	__cuda_parallel_1d__(size){
			dx[idx] = idx;
			dy[idx] = idx;
			dB[idx] = (unsigned char)b;
			dG[idx] = (unsigned char)g;
			dR[idx] = (unsigned char)r;
	}
}



void run(const char* filename,int NumBlocks,int NumThreads){
	// ________________  讀取 .mp4___________________________
	cv::VideoCapture cap(filename);                          		 // 輸入影片檔名
	int VideoFrameH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);            // 影片高
	int VideoFrameW = cap.get(cv::CAP_PROP_FRAME_WIDTH);			 // 影片寬
	double fps = cap.get(cv::CAP_PROP_FPS);    						 // 取得該影片 FPS 資訊
	int size = VideoFrameH*VideoFrameW*3;
	size_t bytes = size*sizeof(unsigned char);  // 計算需要傳輸 的 bytes = H x W x C x (uchar) 

	if(cap.isOpened()){
		// 定義指標 + 配置空間
		cv::Mat input_frame;
		cv::Mat output_frame;
		// 對角線 pixel 數
		int smallHW = min(VideoFrameW,VideoFrameH);
		

		// CPU
		unsigned char* hptr; // CPU input_frame
		


		// GPU
		unsigned char* dptr; // GPU
		int* dx;   // GPU x通道
		int* dy;   // GPU y通道
		unsigned char* dB;   // GPU B通道
		unsigned char* dG;   // GPU G通道
		unsigned char* dR;   // GPU R通道

		
		unsigned char* hptr2 = (unsigned char *)malloc(bytes); // CPU output_frame

		cudaMalloc(&dptr,bytes);
		cudaMalloc(&dx,smallHW*sizeof(int));
		cudaMalloc(&dy,smallHW*sizeof(int));
		cudaMalloc(&dB,smallHW*sizeof(unsigned char));
		cudaMalloc(&dG,smallHW*sizeof(unsigned char));
		cudaMalloc(&dR,smallHW*sizeof(unsigned char));
		


		chrono::steady_clock::time_point t1;
		chrono::steady_clock::time_point t2;
		int frameIdx = 1;

	
		
		// ________________ 掃每一禎 ______________________
		while(cap.read(input_frame)){
			int r = rand();
			int g = rand();
			int b = rand();
			t1 = chrono::steady_clock::now();
			{
				hptr = input_frame.data;  //存到指標上


				cudaMemcpy(dptr,hptr,bytes,cudaMemcpyHostToDevice);
				//===============================================
				// 再 GPU 上配置對角線
				setGPUChannels<<<NumBlocks,NumThreads >>>(dx,dy,dB,dG,dR,smallHW,b,g,r);
				cuda_useful::setBGRs<<< NumBlocks,NumThreads >>>(dptr,dx,dy,VideoFrameW,dB,dG,dR,smallHW);
				//=================================================
				cudaMemcpy(hptr2,dptr,bytes,cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize(); //與主程式同步
				output_frame = cv::Mat(VideoFrameH,VideoFrameW,CV_8UC3,hptr2); // 指標變成 cv::Mat
			} 
			t2 = chrono::steady_clock::now();
			//cout << "=================================================================================================" << endl;
			cout << "f-" << frameIdx <<  " : "  << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " ms" <<"["  <<  chrono::duration_cast<chrono::microseconds>(t2-t1).count() << " mus]"  << endl; 
			

			// 顯示
			cv::imshow("input_frame", input_frame);
			cv::imshow("output_frame",output_frame);
			if(cv::waitKey(1) == 27){ 
		        cout << "Esc !! " << endl; 
		        break; 
		    }//endif
		    frameIdx++;
		}//end_while
		
		// 釋放記憶體
		cudaFree(dptr);
		free(hptr2);

	}//endif
}




// ctypes code 
extern "C"{
	#define DLLEXPORT __declspec(dllexport)
	DLLEXPORT int cuda_run(char* filename,int NumBlocks,int NumThreads){
		run(filename,NumBlocks,NumThreads);
		return 0;
	}
}


int main(int args,char* argv[]){
	if(args != 5){
		cout << "=====================================================================================\n" ;
		cout << "openvino-init : \"C:\\Program Files (x86)\\IntelSWTools\\openvino\\bin\\setupvars.bat\"\n";
		cout << "=====================================================================================\n" ;
		cout << "--run [videoname] [NumBlocks] [NumThreads]" << endl;
		cout << "=====================================================================================\n";
		exit(1);
	}else{
		if(string(argv[1]) == "--run"){
			run(argv[2],atoi(argv[3]),atoi(argv[4]));
		}//endif
	}
	return 0;
}//end_main





