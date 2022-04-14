CC = g++
#opt-in to caching of global memory accesses in L1 via the -Xptxas -dlcm=ca option to nvcc
#intermediate files, using --keep for nvcc
#for PTX codes, using --ptx for nvcc
NVCC = nvcc
main:main.cu
	$(NVCC)  main.cu -o main
test: test.cpp
	$(CC) test.cpp -o test
clean:
	rm -rf main a.out