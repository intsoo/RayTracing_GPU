CC=nvcc 

RTTNW.out:  mkClockMeasure.h  mkCuda.h mkPpm.h main.cu 
	$(CC) -o gpu.out main.cu
	$ ./gpu.out
	$ mv *.ppm ../img

clean:
	rm -rf gpu.out
	rm *.ppm
