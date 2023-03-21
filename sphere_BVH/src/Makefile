CC=nvcc 

RTTNW.out:  mkClockMeasure.h  mkCuda.h main.cu 
	$(CC) -G -g --expt-relaxed-constexpr -o RTTNW.out main.cu

clean:
	rm -rf RTTNW.out img.ppm
