#!/bin/bash

# Compile Verilog code using Verilator
verilator --cc ahb_ai_interface.v  transformer_top.v --top-module ahb_ai_interface --trace

# Compile C++ test bench
mkdir -p obj_dir
verilator --cc --exe --trace ahb_ai_interface.v transformer_top.v test_bench.cpp
make -C obj_dir -f Vahb_ai_interface.mk
obj_dir/Vahb_ai_interface

# View simulation results
gtkwave ahb_ai_interface.vcd

