# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../../crucible_codegen
import std/tables
import ../ir/cirv2_lang
import ../../builtins/builtins_decl_helpers

# ############################################################
#
#            Target registry
#
# ############################################################

genTargets:
  target msl:
    kind unrollDirective "#pragma clang loop unroll(full)", braceSameLine true
    vector:
      elems i32, u32
      lanes 2, 3, 4

  target cuda:
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector ()
    #
    math abs 32 "fabsf", abs 64 "fabs", exp2 32 "exp2f"

  target opencl:
    kind unrollDirective "#pragma unroll", braceSameLine true
    vector:
      elems void, bool, i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
      lanes 2, 3, 4, 8, 16
    math max 32 "fmax", max 64 "fmax", min 32 "fmin", min 64 "fmin",
      abs 32 "fabs", abs 64 "fabs"

  target glsl:
    kind unrollDirective "[[unroll]]", braceSameLine true,
      targetExprKinds {eNil, eAt, ePat, eDeref, eAddr}
    vector:
      elems f16 "f16vec", f32 "vec", f64 "dvec", i32 "ivec", u32 "uvec", bool "bvec"
      lanes 2, 3, 4

  target wgsl:
    math rsqrt 32 "inverseSqrt",
      # <math.h> imports under their C names (in 32 bits, also after demotion)
      logf 32 "log", log 32 "log", expf 32 "exp", exp 32 "exp", sqrtf 32 "sqrt",
      sinf 32 "sin", cosf 32 "cos", tanf 32 "tan", asinf 32 "asin", acosf 32 "acos",
      atanf 32 "atan", fabsf 32 "abs", fabs 32 "abs", truncf 32 "trunc",
      floorf 32 "floor", ceilf 32 "ceil", log2f 32 "log2", exp2f 32 "exp2"
    kind unrollDirective "// {.unroll.} requested but not honored on WebGPU", braceSameLine true,
      targetExprKinds {eNil, eAt, ePat, eTernary, eFloatLit}
    vector:
      elems void "vec", bool "vec", i32 "vec", u32 "vec", f16 "vec", f32 "vec"
      lanes 2, 3, 4
      nameForm vnfGenericElem

  target hlsl:
    kind unrollDirective "[unroll]", braceSameLine false, foldsIteElse true,
      emitOperandRendersExpr true, targetExprKinds {eNil, eAt, ePat, eDeref, eAddr}
    vector:
      elems bool "bool", i32 "int", u32 "uint", f32 "float"
      lanes 2, 3, 4

  target cpu:
    kind unrollDirective "", braceSameLine true, targetExprKinds {eStrLit}
    vector ()
