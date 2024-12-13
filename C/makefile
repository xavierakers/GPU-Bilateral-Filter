# Compiler
CC := gcc

# Compiler flags
CC_FLAGS := -O3 -fPIC -shared

C_SRC_FILES := bilatColorVid.c

# Output shared objects
C_SO_FILES := $(C_SRC_FILES:.c=.so)

# Default target
all: $(C_SO_FILES)

# Rule to compile C .c files into shared libraries (.so)
%.so: %.c
	$(CC) $(CC_FLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(CUDA_SO_FILES)
	rm -f $(C_SO_FILES)
