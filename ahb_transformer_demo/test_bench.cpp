#include "Vahb_ai_interface.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    // Create the device under test (DUT)
    Vahb_ai_interface* dut = new Vahb_ai_interface;

    // Open the VCD trace file
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("ahb_ai_interface.vcd");

    // Initialize the DUT
    dut->ahb_hclk = 0;
    dut->ahb_hresetn = 0;
    dut->ahb_haddr = 0;
    dut->ahb_hburst = 0;
    dut->ahb_hprot = 0;
    dut->ahb_hsel = 0;
    dut->ahb_htrans = 0;
    dut->ahb_hwdata = 0;
    dut->ahb_hwrite = 0;
    dut->clk = 0;
    dut->reset_n = 0;

    // Reset the DUT
    dut->ahb_hresetn = 1;
    dut->reset_n = 1;

    // Simulate the DUT
    for (int i = 0; i < 100; i++) {
        dut->ahb_hclk = !dut->ahb_hclk;
        dut->clk = !dut->clk;
        dut->eval();
        tfp->dump(i);

        // Write the weights and biases to the DUT
        dut->ahb_hsel = 1;
        dut->ahb_hwrite = 1;
        dut->ahb_haddr = 0x0000;
        dut->ahb_hwdata = 0x12345678;
        dut->eval();
        dut->ahb_haddr = 0x0004;
        dut->ahb_hwdata = 0x87654321;
        dut->eval();
        dut->ahb_haddr = 0x0008;
        dut->ahb_hwdata = 0xabcdef01;
        dut->eval();
        dut->ahb_haddr = 0x000C;
        dut->ahb_hwdata = 0x10111213;
        dut->eval();
        dut->ahb_haddr = 0x0010;
        dut->ahb_hwdata = 0x1;
        dut->eval();
        dut->ahb_hsel = 0;
        dut->ahb_hwrite = 0;

        // Check the output
        if (dut->done) {
            printf("Output: %08x\n", dut->ahb_hrdata);
        }
    }

    // Close the VCD trace file and clean up
    tfp->close();
    delete tfp;
    delete dut;
    return 0;
}
