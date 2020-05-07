nvcc -O3 -o hello hello.cu ^
-I"." ^
-I"C:\Program Files (x86)\IntelSWTools\openvino\opencv\include" ^
-L"C:\Program Files (x86)\IntelSWTools\openvino\opencv\lib" ^
-l"opencv_highgui430" -l"opencv_video430" -l"opencv_videoio430" -l"opencv_core430" -l"opencv_imgproc430" -l"opencv_dnn430"