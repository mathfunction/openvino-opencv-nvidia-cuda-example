import ctypes
import sys



if __name__ == "__main__":
	cxx = ctypes.cdll.LoadLibrary("./assignBGRs.dll")
	cxx.cuda_run.argstype = [ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
	cxx.cuda_run.restype = ctypes.c_int
	cxx.cuda_run(sys.argv[1].encode('utf-8'),1,100)