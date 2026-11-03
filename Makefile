CXX := g++
MPICXX := mpic++
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=sm_75 --fmad=false
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
OMPFLAGS := -fopenmp

BUILD := build

.PHONY: all serial openmp mpi cuda test clean

all: serial openmp mpi

serial: $(BUILD)/serial

openmp: $(BUILD)/openmp

mpi: $(BUILD)/mpi

cuda: $(BUILD)/cuda

$(BUILD)/serial: src/serial/main.cpp src/common/*.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ src/serial/main.cpp

$(BUILD)/openmp: src/openmp/main.cpp src/common/*.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ src/openmp/main.cpp

$(BUILD)/mpi: src/mpi/main.cpp src/common/*.hpp
	@mkdir -p $(BUILD)
	$(MPICXX) $(CXXFLAGS) -o $@ src/mpi/main.cpp

$(BUILD)/cuda: src/cuda/main.cu src/common/*.hpp
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ src/cuda/main.cu

$(BUILD)/test_core: tests/test_core.cpp src/common/*.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_core.cpp

test: $(BUILD)/test_core
	./$(BUILD)/test_core

clean:
	rm -rf $(BUILD)
