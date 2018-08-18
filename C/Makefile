HEADERS = neuralNetwork.h utils.h
OBJECTS = neuralNetwork.o main.o utils.o

default: main

%.o: %.c $(HEADERS)
	gcc -Wall -I/home/teleportex/gsl/include -c $< -o $@

main: $(OBJECTS)
	gcc -L/home/teleportex/gsl/lib $(OBJECTS)  -lgsl -lgslcblas -lm -o $@

clean:
	-rm -f $(OBJECTS)
	-rm -f main
run: main
	./main
