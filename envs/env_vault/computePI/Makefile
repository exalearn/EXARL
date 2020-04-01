.PHONY: default

# Use the following for building on Summit
# Replace the include path with your build paths, can be found by mpicc -show (mpich) or -showme (openmpi)
MPICC = $(shell mpicc -show)
LIBNAME = libcomputePI.so

.PHONY: build
build: ${LIBNAME}
${LIBNAME}: computePI.c
	${MPICC} -fPIC -shared -o $@ $<

.PHONY: clean
clean:
	${RM} -r ${LIBNAME} *.pyc __pycache__
