CC = g++
#opt-in to caching of global memory accesses in L1 via the -Xptxas -dlcm=ca option to nvcc
#intermediate files, using --keep for nvcc
#for PTX codes, using --ptx for nvcc
NVCC = nvcc

CFLAGS = -c -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg
EXEFLAG = -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2

main:main.cu Util.o
	$(NVCC)  main.cu  Util.o -o main
Util.o : Util.cpp Util.h
	$(CC) $(CFLAGS) Util.cpp -o Util.o
test: test.cpp
	$(CC) test.cpp -o test
clean:
	rm -rf ./*.o