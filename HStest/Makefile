CXX = g++
CXXFLAGS = -std=c++11 -Wall
LDFLAGS = -lglfw -lGLEW -lGL -ldl -lglm

all: main

main: main.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f main