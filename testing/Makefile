
CULA_PATH = /home/feel/documents/informatics/programming/cuda/lib/cula/culaR12
CUSPLA_PATH = /home/feel/documents/informatics/programming/cuda/dev/cuspla/trunk/cuspla



CC = nvcc

LIB = -L ${CULA_PATH}/lib/ -lcula
LIB += -ldl -lcppunit -lcublas -lcudart

INCL = -I ../
INCL += -I ${CULA_PATH}/include
INCL += -I ${CUSPLA_PATH}/include

CFLAGS_COMMON = --disable-warnings -arch=sm_11 -Xcompiler -fmessage-length=0
CFLAGS = -O2 

SYMS = -DVERBOSE #-DVVERBOSE

OUT1 = test_arnoldi
FILE1 = test_arnoldi.cu



OUT_ALL = $(OUT1)


all: $(OUT_ALL)


test_arnoldi:
	$(CC) $(SYMS) $(CFLAGS_COMMON) $(CFLAGS) $(LIB) $(INCL) -o $(OUT1) $(FILE1)

clean: 
	rm $(OUT_ALL)

