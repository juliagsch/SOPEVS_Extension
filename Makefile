PYTHON_INCLUDES := $(shell python3-config --includes)
PYTHON_LIBS := -L/opt/homebrew/Cellar/python@3.11/3.11.13/Frameworks/Python.framework/Versions/3.11/lib -lpython3.11

all: sim

sim:
	g++ -std=c++14 -O2 run_simulation.cc cheby.cc simulate_system.cc common.cc ev.cc -o bin/sim $(PYTHON_INCLUDES) $(PYTHON_LIBS)

debug: debug_sim

debug_sim:
	g++ -std=c++14 -O0 -ggdb -D DEBUG run_simulation.cc cheby.cc simulate_system.cc common.cc ev.cc -o bin/debug/sim $(PYTHON_INCLUDES) $(PYTHON_LIBS)

clean:
	rm bin/sim