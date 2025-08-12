#Sources and targets
SOURCES = sim_main.cpp top.sv Counter.sv
SIMULATOR = ./obj_dir/Vtop
DUMPFILE = dump.vcd
CLEANFILES = obj_dir $(DUMPFILE)
vpath %.sv ../src

# Commands
VERILATOR = verilator

# Flags for commands
VERILATOR_FLAGS = --cc --exe --build -j 0 -Wall --trace
SIMULATOR_FLAGS = +trace

# Rules

.PHONY: all build run clean

all: run;
run: $(DUMPFILE);
build: $(SIMULATOR);

$(SIMULATOR): $(SOURCES)
	$(VERILATOR) $(VERILATOR_FLAGS) $^

$(DUMPFILE): $(SIMULATOR)
	$(SIMULATOR) $(SIMULATOR_FLAGS)

clean:
	-rm -fr $(CLEANFILES)


