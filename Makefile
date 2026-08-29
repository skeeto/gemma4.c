CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS = -lm
WINCC = x86_64-w64-mingw32-gcc

# Apple clang has no -fopenmp driver support, so use the frontend flag with Homebrew's libomp.
ifeq ($(shell uname -s), Darwin)
LIBOMP = $(shell brew --prefix libomp 2>/dev/null)
CFLAGS = -std=c11 -O3 -Wall -Wextra -mcpu=native -Xclang -fopenmp -I$(LIBOMP)/include
LDFLAGS = -lm -L$(LIBOMP)/lib -lomp
endif

run: gemma4.c
	$(CC) $(CFLAGS) gemma4.c -o run $(LDFLAGS)

server: server.c gemma4.c
	$(CC) $(CFLAGS) server.c -o server $(LDFLAGS)

run-e4b: gemma4.c
	$(CC) $(CFLAGS) -DGEMMA4_E4B gemma4.c -o run-e4b $(LDFLAGS)

server-e4b: server.c gemma4.c
	$(CC) $(CFLAGS) -DGEMMA4_E4B server.c -o server-e4b $(LDFLAGS)

win64: gemma4.c win.c win.h
	$(WINCC) -std=c11 -O3 -Wall -Wextra -march=native -fopenmp -static -D_WIN32 gemma4.c win.c -o run.exe -lm -lshell32

.PHONY: clean win64
clean:
	rm -f run run.exe server run-e4b server-e4b
