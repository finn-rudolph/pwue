all:
	g++ -std=c++20 -Wall -Wextra -Ofast -march=native -msse4 -mavx -mavx2 -o pwue pwue.cpp

clean:
	rm pwue
