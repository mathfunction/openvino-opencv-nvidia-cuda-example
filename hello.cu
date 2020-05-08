/*==========================================================================================
// 這是關於把一個影片輸入，利用 GPU 運算 ，把彩色影片轉成黑白影片

// opencv == H x W x C (BGR)

=============================================================================================*/
#include <chrono>
#include <iostream>
// nvidia cuda library  
#include "npp.h"
#include "cublas.h"
#include "cusparse.h"


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


int main(int args,char* argv[]){
	if(args != 3){
		cout << "=====================================================================================\n" ;
		cout << "openvino-init : \"C:\\Program Files (x86)\\IntelSWTools\\openvino\\bin\\setupvars.bat\"\n";
		cout << "=====================================================================================\n" ;
		cout << "--run [videoname]" << endl;
		exit(1);
	}

	// ________________  讀取 .mp4___________________________
	cv::VideoCapture cap(argv[2]);                          		 // 輸入影片檔名
	int VideoFrameH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);            // 影片高
	int VideoFrameW = cap.get(cv::CAP_PROP_FRAME_WIDTH);			 // 影片寬
	double fps = cap.get(cv::CAP_PROP_FPS);    						 // 取得該影片 FPS 資訊
	int size = VideoFrameH*VideoFrameW*3;
	size_t bytes = size*sizeof(unsigned char);  // 計算需要傳輸 的 bytes = H x W x C x (uchar) 

	if(cap.isOpened()){
		// 定義指標 + 配置空間
		cv::Mat input_frame;
		cv::Mat output_frame;
		unsigned char* hptr; // CPU input_frame
		unsigned char* dptr; // GPU
		unsigned char* hptr2 = (unsigned char *)malloc(bytes); // CPU output_frame
		cudaMalloc(&dptr,bytes);
		chrono::steady_clock::time_point t1;
		chrono::steady_clock::time_point t2;
		int frameIdx = 1;

		// ________________ 掃每一禎 ______________________
		while(cap.read(input_frame)){
			t1 = chrono::steady_clock::now();
			{
				hptr = input_frame.data;  //存到指標上

				cudaMemcpy(dptr,hptr,bytes,cudaMemcpyHostToDevice);
				//===============================================
				// do something on dptr at GPU + CUDA .....

				cuda_useful::setConstantBGR<<< 2,512 >>>(dptr,size,255,255,255);
				//=================================================
				cudaMemcpy(hptr2,dptr,bytes,cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize(); //與主程式同步
				output_frame = cv::Mat(VideoFrameH,VideoFrameW,CV_8UC3,hptr2); // 指標變成 cv::Mat
			} 
			t2 = chrono::steady_clock::now();
			//cout << "=================================================================================================" << endl;
			cout << "[" << frameIdx << "] : "  <<  chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << "ms" << endl; 
			



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

	return 0;
}//end_main