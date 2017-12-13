src     = calcIR.c
exes    = JansenIR.exe
CC      = mpicc
LIBS    = -lxdrfile -lm -lfftw3

all: ${exes}

${exes}: ${src}
	$(CC) $(src) -o $(exes) $(LIBS) -DUSE_DOUBLES

clean:
	rm JansenIR.exe
