ifeq (0,$(MAKELEVEL))
ERROR:
	@echo "ERROR:  Please use toplevel makefile"
endif

INCLUDES = ../nekBone/BP.h BPPrecon.h 
DEPS = $(INCLUDES) \
$(HDRDIR)/timer.hpp \
$(HDRDIR)/mesh.h \
$(HDRDIR)/mesh2D.h \
$(HDRDIR)/mesh3D.h \
$(OGSDIR)/ogs.hpp \
${CORE_DIR}/$(AXHELMDIR)/kernelHelper.cpp \
${CORE_DIR}/setCompilerFlags.hpp \
${CORE_DIR}/drandalloc.hpp \

# types of files we are going to construct rules for
.SUFFIXES: .c .f .cpp

# rule for .c files
.c.o: $(DEPS)
	$(CC) $(CFLAGS) -o $*.o -c $*.c 

.cpp.o: $(DEPS)
	$(CXX) $(CXXFLAGS) -o $*.o -c $*.cpp

.f.o: $(DEPS)
	$(FC) $(CFLAGS) -o $*.o -c $*.f 

# list of objects to be compiled
AOBJS    = \
./main.o \
./driver.o \
../nekBone/setup.o \
../nekBone/solve.o \
../nekBone/nekBone.o \
../gs/ping_pong.o \
../gs/mygs.o \
../gs/gs.o \
../dot/dot.o \
../allred/allred.o \
../bw/bw.o \
../axhelm/axhelm.o \
../axhelm/kernelHelper.o \



# library objects
LOBJS = \
../core/timer.o \
../core/meshSetup.o \
../core/meshBasis.o \
../core/setupAide.o \
./inireader.o

.PHONY: all install clean realclean

all: driver install 

install:
	@mkdir -p $(PREFIX)
	@cp driver.ini $(PREFIX)/driver.ini
	@cd ../bw/kernel; find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \;
	@cd ../dot/kernel; find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \;
	@cp ../gs/*.okl $(PREFIX) 
	@cd ../axhelm/kernel; find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \;
	@cd ../nekBone/kernel; find . -type f -name "*" -exec install -D {} $(PREFIX)/kernel/{} \;
	@cd $(CURDIR)	 
	@mv driver $(PREFIX) 


driver: $(AOBJS) $(LOBJS)
	$(LD)  -o driver $(COBJS) $(AOBJS) $(LOBJS) $(LDFLAGS)

clean:
	@rm -rf *.o
