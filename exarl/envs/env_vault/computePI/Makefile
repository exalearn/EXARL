.PHONY: default

# Use the following for building on Summit
# Replace the include path with your build paths, can be found by mpicc -show (mpich) or -showme (openmpi)
MPICC = $(shell mpicc -show)
LIBNAME = libcomputePI.so

.PHONY: build
build: ${LIBNAME}
${LIBNAME}: computePI.c
	${MPICC} -fPIC -shared -o $@ $<
	# /sw/summit/gcc/8.1.1-cuda10.1.168/bin/gcc -I/ccs/home/vinayr/.conda/envs/exarl_summit/include -I/ccs/home/vinayr/.conda/envs/exarl_summit/include -L/ccs/home/vinayr/.conda/envs/exarl_summit/lib -Wl,-rpath,/ccs/home/vinayr/.conda/envs/exarl_summit/lib -I/ccs/home/vinayr/.conda/envs/exarl_summit/include -L/ccs/home/vinayr/.conda/envs/exarl_summit/lib -lmpi -fPIC -shared -o libcomputePI.so computePI.c

.PHONY: clean
clean:
	${RM} -r ${LIBNAME} *.pyc __pycache__
