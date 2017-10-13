CXX=icpc

GIT_VERSION:=$(shell sh -c './GIT-VERSION-GEN')
COMPILER_VERSION:="$(CXX)-$(shell $(CXX) --version | head -n1 | cut -d' ' -f4)"
BUILD_HOST=$(shell sh -c './BUILD-HOST-GEN')

HFAV_DIR=../../
HFAVROOT?=$(HFAV_DIR)/hfav

HFAV=$(HFAV_DIR)/hfav.py

SSE_CXXFLAGS=-DSSE -xSSE4.2
KNC_CXXFLAGS=-DKNC -mmic
AVX_CXXFLAGS=-DAVX -xAVX
AVX2_CXXFLAGS=-DAVX -xCORE-AVX2
KNL_CXXFLAGS=-DAVX3 -xMIC-AVX512
SKL_CXXFLAGS=-DAVX3 -xCORE-AVX512

COMPILER_INC?=/opt/intel/compilers_and_libraries/linux/include/mic
ARCH_CXXFLAGS=

ifeq ($(ARCH),SSE)
	ARCH_CXXFLAGS=$(SSE_CXXFLAGS)
endif

ifeq ($(ARCH),KNC)
	ARCH_CXXFLAGS=$(KNC_CXXFLAGS)
endif

ifeq ($(ARCH),AVX)
	ARCH_CXXFLAGS=$(AVX_CXXFLAGS)
endif

ifeq ($(ARCH),AVX2)
	ARCH_CXXFLAGS=$(AVX2_CXXFLAGS)
endif

ifeq ($(ARCH),KNL)
	ARCH_CXXFLAGS=$(KNL_CXXFLAGS) -I$(COMPILER_INC)
endif

ifeq ($(ARCH),SKL)
	ARCH_CXXFLAGS=$(KNL_CXXFLAGS)
endif


BASE_CXXFLAGS=-DGIT_VERSION=\"$(GIT_VERSION)\" -DCOMPILER_VERSION=\"$(COMPILER_VERSION)\" \
	      -DBUILD_HOST=\"$(BUILD_HOST)\" \
	      -DDOUBLE \
	      -Wall -wd167 -ggdb \
	      -qopenmp \
	      -lnuma \
          -O3 -ipo -restrict -qopt-report=5 -inline-forceinline \
          -no-prec-div -no-prec-sqrt -fimf-precision=low -fimf-domain-exclusion=15 \
	      -I$(HFAVROOT)/include

ull: run-tile run-gen compare

run-tile: run-tile.cpp pcl-hydro-core.cpp pcl-hydro-vcore.cpp pcl-hydro-util.cpp pcl-hydro-params.cpp vtkfile.cpp timeseries.cpp pcl-hydro.hpp arch.hpp config.c Makefile
	$(CXX) -o $@ run-tile.cpp pcl-hydro-core.cpp pcl-hydro-vcore.cpp pcl-hydro-util.cpp pcl-hydro-params.cpp vtkfile.cpp timeseries.cpp config.c $(BASE_CXXFLAGS) $(ARCH_CXXFLAGS)

hydro2d-x-gen.hpp: hydro2d-x.yaml $(HFAV) $(HFAV_DIR)/hfav/infer.py $(HFAV_DIR)/hfav/dag.py $(HFAV_DIR)/hfav/analyze.py $(HFAV_DIR)/hfav/codegen.py $(HFAV_DIR)/hfav/term.py $(HFAV_DIR)/hfav/c99.py $(HFAV_DIR)/hfav/parse.py
	$(HFAV) $(HFAV_FLAGS) hydro2d-x.yaml

hydro2d-y-gen.hpp: hydro2d-y.yaml $(HFAV) $(HFAV_DIR)/hfav/infer.py $(HFAV_DIR)/hfav/dag.py $(HFAV_DIR)/hfav/analyze.py $(HFAV_DIR)/hfav/codegen.py $(HFAV_DIR)/hfav/term.py $(HFAV_DIR)/hfav/c99.py $(HFAV_DIR)/hfav/parse.py
	$(HFAV) $(HFAV_FLAGS) hydro2d-y.yaml

run-gen: run-tile.cpp pcl-hydro-core.cpp pcl-hydro-vcore.cpp pcl-hydro-util.cpp pcl-hydro-params.cpp vtkfile.cpp timeseries.cpp pcl-hydro.hpp config.c arch.hpp Makefile hydro2d-x-gen.hpp hydro2d-y-gen.hpp
	$(CXX) -o $@ run-tile.cpp pcl-hydro-core.cpp pcl-hydro-vcore.cpp pcl-hydro-util.cpp pcl-hydro-params.cpp vtkfile.cpp timeseries.cpp config.c -I $(HFAVROOT)/include/ $(BASE_CXXFLAGS) $(ARCH_CXXFLAGS) -DUSE_GEN_X -DUSE_GEN_Y -qopt-report=5

compare: compare.cpp timeseries.cpp Makefile
	$(CXX) -o $@ compare.cpp timeseries.cpp $(BASE_CXXFLAGS)

clean:
	rm -rf run-tile compare hydro2d-x-gen.hpp hydro2d-y-gen.hpp *.optrpt
