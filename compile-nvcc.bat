nvcc -O3 -o hello hello.cu ^
-I"C:\Program Files (x86)\IntelSWTools\openvino\opencv\include" ^
-L"C:\Program Files (x86)\IntelSWTools\openvino\opencv\lib" ^
-l"opencv_highgui411" -l"opencv_video411" -l"opencv_videoio411" -l"opencv_core411" -l"opencv_imgproc411" -l"opencv_dnn411"