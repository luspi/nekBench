WS ?= 8

PROJ_DIR=./
CORE_DIR=$(HDRDIR)

ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

.PHONY: all install clean realclean

all: allred install

allred: main.cpp allred.cpp ${CORE_DIR}/timer.cpp ${CORE_DIR}/setCompilerFlags.hpp ${CORE_DIR}/setupAide.cpp
	$(CXX) $(CXXFLAGS) -o allred main.cpp allred.cpp ${CORE_DIR}/timer.cpp ${CORE_DIR}/setupAide.cpp $(linkerFlags) $(LDFLAGS)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(CXX) $(CXXFLAGS) -o $@ -c $<

install:
	@mkdir -p $(PREFIX)/kernel
	@mv allred $(PREFIX)

clean:
	@rm -rf *.o
