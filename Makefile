FLAGS=-std=c++11 -O2

label: locallabel.o
	nvcc $(FLAGS) -o label locallabel.o 

locallabel.o: locallabel.cu unionfind.cu memory.cu kernel.cu stb_image.h image.h
	nvcc $(FLAGS) -c locallabel.cu

clean:
	rm label locallabel.o