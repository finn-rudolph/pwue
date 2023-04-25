all:
	g++ -std=c++20 -pthread -Wall -Wextra -Ofast -march=native -msse4 -mavx -mavx2 -o pwue pwue.cpp

debug:
	g++ -std=c++20 -pthread -Wall -Wextra -g -fsanitize=address,undefined -D_GLIBCXX_DEBUG -o pwue pwue.cpp

clean:
	rm pwue
