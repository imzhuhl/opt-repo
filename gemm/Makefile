MY_IMPL = gemm_1.cpp

ifndef DEBUG
	COMPILE_FLAG += -O2
else
	COMPILE_FLAG += -g -DDEBUG
endif

CXX = g++
COMPILE_FLAG += -std=c++17

check: utils.hpp $(MY_IMPL) check.cpp
	$(CXX) $(COMPILE_FLAG) $(MY_IMPL) check.cpp -o check.out

performance: $(MY_IMPL) performance.cpp
	$(CXX) -O2 $(MY_IMPL) performance.cpp -o performance.out	

clean:
	rm *.x
