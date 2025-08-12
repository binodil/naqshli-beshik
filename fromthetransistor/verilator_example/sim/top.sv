// The top module in SystemVerilog
module top (    
    input logic clk,
    input logic rst_n
);
  // The configuration of the DUT
  localparam int unsigned DUT_WIDTH = 4;

  // The DUT
  logic                 dut_enb;
  // verilator lint_off UNUSEDSIGNAL
  logic [DUT_WIDTH-1:0] dut_count;
  // verilator lint_on UNUSEDSIGNAL
  logic                 dut_carryout;
  Counter #(
      .WIDTH(DUT_WIDTH)
  ) dut (
      .clk     (clk),
      .rst_n   (rst_n),
      .enb     (dut_enb),
      .count   (dut_count),
      .carryout(dut_carryout)
  );

  // The test scenario
  typedef enum logic [1:0] {
    STATE_INITIAL,
    STATE_RUNNING,
    STATE_FINISHED
  } state_e;
  state_e state;
  always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
      state <= STATE_INITIAL;
    end else if (state == STATE_INITIAL) begin
      state <= STATE_RUNNING;
    end else if (state == STATE_RUNNING && dut_carryout) begin
      state <= STATE_FINISHED;
    end else if (state == STATE_FINISHED) begin
      $finish();
    end
  end
  assign dut_enb = (state == STATE_RUNNING);

  // Enable tracing if `+trace` argument specified
  initial begin
    if ($test$plusargs("trace") != 0) begin
      $display("Start tracing to dump.vcd...");
      $dumpfile("dump.vcd");
      $dumpvars();
    end
  end

endmodule

