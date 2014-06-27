CC		  := icc
LD      := icc
CFLAGS  := -mmic -vec-report3 -fopenmp -MP -MD -c  
LDFLAGS := -mmic -vec-report3 -fopenmp -MP -MD 
SOURCES	  := $(shell echo include/common/*.cpp include/convnet/*.cpp)
HEADERS	  := $(shell echo include/common/*.h include/convnet/*.h)
OBJECTS	  := $(SOURCES:.cpp=.o)
TESTS   := $(shell echo test/*.cpp)

all: $(SOURCES) $(TESTS)

$(TESTS) : $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $@ -o $@.o

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
	
clean:
	rm $(OBJECTS)
