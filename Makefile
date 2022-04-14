CC = g++
NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G --ptxas-options=-v -lineinfo -Xcompiler -rdynamic -I ~/cudaToolkit/cub-1.8.0/
objdir = ./objs/
CFLAGS = -c -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg
EXEFLAG = -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2

objfile = $(objdir)Util.o $(objdir)IO.o $(objdir)Graph.o
pre_GSI.exe: $(objfile) main/run.cpp
	$(NVCC) $(EXEFLAG) -o pre_GSI.exe main/run.cpp $(objfile)

$(objdir)Util.o: util/Util.cpp util/Util.h
	$(CC) $(CFLAGS) util/Util.cpp -o $(objdir)Util.o

$(objdir)Graph.o: graph/Graph.cpp graph/Graph.h
	$(CC) $(CFLAGS) graph/Graph.cpp -o $(objdir)Graph.o

$(objdir)IO.o: io/IO.cpp io/IO.h
	$(CC) $(CFLAGS) io/IO.cpp -o $(objdir)IO.o
clean:
	rm -f $(objdir)*
