MY_IMPL = gemm_1.cpp

ifndef DEBUG
	COMPILE_FLAG += -O2
else
	COMPILE_FLAG += -g -DDEBUG
endif

ifdef CHECK
	COMPILE_FLAG += -DCHECK
endif

CXX = g++
COMPILE_FLAG += -std=c++17

check: utils.hpp $(MY_IMPL) check.cpp
	$(CXX) $(COMPILE_FLAG) $(MY_IMPL) check.cpp -o check.out

performance: $(MY_IMPL) performance.cpp
	$(CXX) $(COMPILE_FLAG) $(MY_IMPL) performance.cpp -o performance.out	

sbgemm:
	$(CXX) $(COMPILE_FLAG) -march=armv8.6-a sbgemm.cpp -o sbgemm.out

clean:
	rm *.x
