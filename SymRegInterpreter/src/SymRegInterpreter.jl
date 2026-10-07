"""
    SymRegInterpreter

Representation and fast evaluation of symbolic regression models: a [`Model`](@ref)
is an expression or a linear program of [`Instruction`](@ref)s, which
[`interpret_vec`](@ref), [`interpret_grad!`](@ref) and [`interpret_jac!`](@ref)
evaluate with reverse-mode derivatives and without allocations.
"""
module SymRegInterpreter

using PrecompileTools: @setup_workload, @compile_workload
using ForwardDiff
using PreallocationTools
using Random
using Printf
import NaNMath: NaNMath

export Model, ExprModel, CodeModel, n_param,
    interpret, interpret_vec, interpret_vec!, interpret_grad!, interpret_jac, interpret_jac!

public Instruction, Opcode,
    STOP, ADD, SUB, MUL, DIV, INV, LOG, LOG10, LOGABS, EXP, POW, POWCONST, POWABS,
    NEG, ABS, SIGN, SQR, SQRT, SQRTABS, SIN, SINH, ASIN, TAN, TANH, COS, COSH, CONSTANT,
    PARAM, VAR, SCALEDVAR,
    terminal_opcodes, unary_opcodes, binary_opcodes, opcode, degree,
    create_const_instruction, create_param_instruction, create_var_instruction,
    create_scaledvar_instruction, print_program,
    code, expr, core_expr, replace_param_with_const, differentiate,
    to_logspace_x_model, to_logspace_y_model, to_logspace_xy_model,
    with_fit_buffer_cache, reset_fit_buffer_cache!, EltypeMismatchError

include("buffers.jl")
include("code.jl")
include("model.jl")
include("differentiate.jl")
include("interpreter.jl")

include("precompile.jl")

end
