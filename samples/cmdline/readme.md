# cmdline

This is a swift app with random experiments in swift.

Mainly to checkout the apple hardware acceleration provided by accelerate
and to look at the mlcoretools for ane models also there is some neat profiling within
the xcode ecosystem.


### Programs
1. vDSP_mul program, generates about 1.6GB of random data in the form of two vectors, it then uses the SIMD instruction multiply the data together really fast. I added some timing and messages to the program to see how long each step takes and to compare to for loop CPU multiply. 

### todo
1. I want to get an FFT working on accelerate
2. Explore BLAS
3. Explore vForce
4. Explore vImage
5. Explore BNNS