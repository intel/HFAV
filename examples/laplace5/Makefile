HFAV_DIR=../../
HFAVROOT?=$(HFAV_DIR)/hfav

HFAV=$(HFAV_DIR)/hfav.py

all: reference generated

reference: laplace5-test.cpp
	icpc -o reference laplace5-test.cpp -fopenmp -restrict -std=c++11 -xHost

laplace5-gen.hpp: $(HFAV) laplace5.yaml
	$(HFAV) laplace5.yaml

generated: laplace5-test.cpp laplace5-gen.hpp
	icpc -o generated laplace5-test.cpp -fopenmp -restrict -DUSE_GEN -std=c++11 -xHost -I$(HFAVROOT)/include

clean:
	rm -rf generated reference laplace5-gen.hpp
