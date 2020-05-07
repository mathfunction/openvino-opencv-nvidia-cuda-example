/*==========================================================================================

// openvino option :
	"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"


=============================================================================================*/

#include <iostream>
// nvidia cuda library  
#include "npp.h"


// intel-openvino-opencv
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>

using namespace std;


int main(int args,char* argv[]){
	if(args != 3){
		cout << "openvino-init : \"C:\\Program Files (x86)\\IntelSWTools\\openvino\\bin\\setupvars.bat\" "
		cout << "--run [videoname]" << endl;
		exit(1);
	}

	// ________________  讀取 .mp4___________________________
	cv::VideoCapture cap(argv[2]);
	int VideoFrameH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int VideoFrameW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	double fps = cap.get(cv::CAP_PROP_FPS);
	int FrameSize = VideoFrameH * VideoFrameW;
	size_t bytes = FrameSize*3*sizeof(unsigned char);

	

	if(cap.isOpened()){
		//________________ 定義指標_________________________ 
		cv::Mat frame;
		cv::Mat frame2;
		unsigned char* hptr; // CPU 處理前
		unsigned char* dptr; // GPU
		unsigned char* hptr2 = (unsigned char *)malloc(bytes); // CPU 處理後
		cudaMalloc(&dptr,bytes);
		
		// ________________ 掃每一禎 ______________________
		while(cap.read(frame)){
			hptr = frame.data;
			cudaMemcpy(dptr,hptr,bytes,cudaMemcpyHostToDevice);
			// _____________ CUDA HANDLE BLOCK ________________________________________









			//_________________________________________________________________________
			cudaMemcpy(hptr2,dptr,bytes,cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			frame2 = cv::Mat(VideoFrameH,VideoFrameW,CV_8UC3,hptr2);

			cv::imshow("frame", frame);
			cv::imshow("frame2",frame2);
			if(cv::waitKey(1) == 27){ 
		        cout << "Esc !! " << endl; 
		        break; 
		    }//endif
		}//end_while

		cudaFree(dptr);
		free(hptr2);

	}//endif

	

	return 0;
}//end_main