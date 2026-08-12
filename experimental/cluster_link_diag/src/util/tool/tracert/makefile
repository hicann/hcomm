CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
THIRDLIB_ROOT ?= /usr/local/third_lib
INCLUDES := -I. -I.. -I../.. -I$(THIRDLIB_ROOT)/include
OUT_DIR := build

.PHONY: all clean

all: $(OUT_DIR)/proc_res

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/proc_res: proc_res.cpp | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) proc_res.cpp -o $@

clean:
	rm -f $(OUT_DIR)/proc_res
