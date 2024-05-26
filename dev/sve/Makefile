#CFLAGS += -march=armv8-a+sve
CC ?= gcc
LDLBIS ?= -lm

MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

.PRECIOUS: %.o
.SUFFIXES:

all: attention.o encoder.o exp.o gelu.o gpt2_model.o layernorm.o \
	log2.o matmul.o residual.o softmax+crossentropy.o tanh.o utils+tokenizer.o

%.o: %.c gpt2_sve.h gpt2_model.h
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f *.o
