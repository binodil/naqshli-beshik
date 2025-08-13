// candy is a low performance RISC-V processor





//module decode (
//  input [31:0] ins,
//  output [31:0] opcode,

//);
//  assign opcode = ins[16:0];
//endmodule


module candy (
  input clk, resetn
);


  always @(posedge clk) begin
  end
endmodule

module testbench;
  reg clk;

  initial begin
    $display("work");
    clk = 0;
  end
  
  always
    #5 clk = !clk;

  candy c (
    .clk (clk),
    .resetn (1'b0)
  );
  
  initial begin
    #100 $finish;

  end
endmodule

