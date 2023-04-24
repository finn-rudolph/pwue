all:
	g++ -std=c++20 -Wall -Wextra -Ofast -march=native -msse4 -mavx -mavx2 -o pwue pwue.cpp

debug:
	g++ -std=c++20 -Wall -Wextra -g -fsanitize=address,undefined -D_GLIBCXX_DEBUG -o pwue pwue.cpp

clean:
	rm pwue
