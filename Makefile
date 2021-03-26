CC = nvcc
DEBUG = -g -G

all: mult add


mult: mult.cu persistent_mult.cu common.o
	$(CC) $(DEBUG) $^ -o $@

add: add.cu persistent_add.cu common.o
	$(CC) $(DEBUG) $^ -o $@

common.o: common.cu
	$(CC) -c $^ -o $@ $(DEBUG)

clean:
	rm mult
