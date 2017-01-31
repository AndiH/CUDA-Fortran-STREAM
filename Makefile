FC = pgfortran
FCFLAGS += -Mcuda -Minfo=all

ifdef DEBUG
FCFLAGS += -g
endif

SOURCE = stream.F90

.PHONY: all
all: stream.bin

stream.bin: Makefile $(SOURCE)
	${FC} -o $@ ${FCFLAGS} $(SOURCE)

.PHONY: clean
clean:
	rm *.bin
	rm *.mod

.PHONY: run
run: stream.bin
	./stream.bin
