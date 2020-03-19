.PHONY: default
default: build clean

MPICC = mpicc
LIBNAME = libcomputePI.so
.PHONY: build
build: ${LIBNAME}
${LIBNAME}: computePI.c
	${MPICC} -fPIC -shared -o $@ $<

.PHONY: clean
clean:
	${RM} -r ${LIBNAME} *.pyc __pycache__
