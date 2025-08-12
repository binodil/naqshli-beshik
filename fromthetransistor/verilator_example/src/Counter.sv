module Counter #(
    parameter int unsigned WIDTH = 1
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             enb,
    output logic [WIDTH-1:0] count,
    output logic             carryout
);

  always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
      count <= 0;
    end else if (enb) begin
      count <= count + 1;
    end
  end
  assign carryout = (enb && count == '1);

endmodule
