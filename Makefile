TARGET ?= seq_opt-none
COMPILER ?= gcc

include config/make.config.$(COMPILER)

C_SOURCES += constants.c grid.c main.c pml.c
EXE = main_$(TARGET)_$(COMPILER)

C_OBJECTS=$(C_SOURCES:.c=.o)

%.o: %.cc
	$(CC) $(CFLAGS) $(HPCTOOLKITFLAGS) -c $< -o $@

all: $(EXE)

$(EXE): $(C_OBJECTS) $(DEVICE_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(HPCTOOLKITFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -rf $(EXE) *.o targets/*/*.o
