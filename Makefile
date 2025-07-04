PYTHON_INCLUDES := $(shell python3-config --includes)
PYTHON_LIBS := $(shell python3-config --ldflags) -lpython3.12

all: sim

sim:
	g++ -std=c++14 -O2 run_simulation.cc cheby.cc simulate_system.cc common.cc ev.cc -o bin/sim $(PYTHON_INCLUDES) $(PYTHON_LIBS)

debug: debug_sim

debug_sim:
	g++ -std=c++14 -O0 -ggdb -D DEBUG run_simulation.cc cheby.cc simulate_system.cc common.cc ev.cc -o bin/debug/sim $(PYTHON_INCLUDES) $(PYTHON_LIBS)

clean:
	rm bin/sim bin/snc_lolp bin/snc_eue bin/debug/sim bin/debug/snc_lolp bin/debug/snc_eue