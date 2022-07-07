MY_IMPL = gemm_6.cpp

ifndef DEBUG
	COMPILE_FLAG += -O2
else
	COMPILE_FLAG += -g -DDEBUG
endif

CXX = g++
COMPILE_FLAG += -std=c++17

all: utils.hpp $(MY_IMPL) main.cpp
	$(CXX) $(COMPILE_FLAG) $(MY_IMPL) main.cpp -o main.out

clean:
	rm *.x
