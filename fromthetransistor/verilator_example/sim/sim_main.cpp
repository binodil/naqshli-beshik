nclude <memory>
#include "verilated.h"
#include "Vtop.h"

int main(int argc, char **argv) {
    // Create a VerilatedContext
    std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
    contextp->commandArgs(argc, argv);

    // Create the Verilated model of the top module
    std::unique_ptr<Vtop> top{new Vtop{contextp.get(), "top"}};

    // Setup the context
    contextp->timeunit(-9);  // Set timeunit to 1ns
    contextp->timeprecision(-9);  // Set timeprecision to 1ns
    contextp->traceEverOn(true);  // Allow traces

    // Run
    top->clk = 0;
    top->rst_n = !0;
    while (!contextp->gotFinish()) {
        top->clk = !(top->clk);
        if (!top->clk) {
            top->rst_n = !(contextp->time() > 0 && contextp->time() < 20);
        }
        top->eval();
        contextp->timeInc(5);
    }

    // Finalize
    top->final();
    return 0;
}

