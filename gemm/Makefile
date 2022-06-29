CXX = ~/opt/gcc-12.1.0/bin/g++
COMPILE_FLAG = -std=c++17

main: utils.hpp main.cpp
	$(CXX) $(COMPILE_FLAG) main.cpp -o main.x


debug: utils.hpp main.cpp
	$(CXX) -g -DDEBUG $(COMPILE_FLAG) main.cpp -o main.x


clean:
	rm *.x
