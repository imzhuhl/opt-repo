MY_IMPL = gemm_4.cpp

ifndef DEBUG
	COMPILE_FLAG += -O2
else
	COMPILE_FLAG += -g -DDEBUG
endif

CXX = g++
COMPILE_FLAG += -std=c++17

all: utils.hpp $(MY_IMPL) main.cpp
	$(CXX) $(COMPILE_FLAG) $(MY_IMPL) main.cpp -o main.x

clean:
	rm *.x
