# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import ../ir/cirv2_lang
import ../frontend/nim_to_crucible
import ../passes/pass_helpers
import ../passes/passes_lowering_wgsl
import std/[macros, math, sets, strutils, tables]
import ./print_common
import ./targets_registry
import ../../builtins/builtins_gpu_types
import ../../crucible_codegen
import ../../builtins/builtins_decl_helpers

# ############################################################
#
#            WGSL printer
#
# ############################################################
{.experimental: "codeReordering".}

const printerTarget = "wgsl"

type PrintCtx = object of WalkCtx
  ## Per-target walk context. The shared walk procs instantiate against
  ## the derived type and the overlay procs below resolve per target.
  m: CirModule
  ## Module in flight.
  atomics: HashSet[string]
  ## `proc\x1fsymbol` of the buffers and pointer parameters an atomic
  ## builtin reaches: their elements print as `atomic<T>`.

# ═══════════════════════════════════════════════════════════════════════
#  Target tables
# ═══════════════════════════════════════════════════════════════════════

const wgslVocab = BuiltinVocab(
  reductionText: makeTexts([
    (cbkReduceAdd, "subgroupAdd($1)"),
    (cbkReduceMax, "subgroupMax($1)")]),
  shuffleText: makeTexts([
    # shuffle index parameter is `u32`
    (cbkSimdShuffle, "subgroupShuffle($1, u32($2))"),
    (cbkSimdShuffleXor, "subgroupShuffleXor($1, u32($2))")]))

const wgslEmit = TargetEmit(
  vocab: wgslVocab,
  functionalCtor: true,
  ctorOpen: "(",
  ctorClose: ")",
  ctorCast: false,
  ctorZero: "$1(0u)",
  aconstrCasts: false,
  dotDerefParens: false,
  vectorDots: true,
  noPointerValues: (spaces: {}, text: ""),
  noPointerRebind: "",
  prefixGlobalFn: "",
  prefixInline: "",
  prefixDevice: "")

const wgslModuleShape = ModuleShape(
  spaces: {caThreadgroup, caConstant},
  structTypedef: false,
  structPadding: "_padding: u32",
  structFieldTerm: ',',
  structClose: "}",
  fwdDecls: false,
  moduleOrderProcs: false,
  globalFnSep: "\n",
  customProcHead: true,
  printsRefBlocks: false,
  printsParamBindings: true,
  printsReduceTemplates: false)

# WGSL has no infinity or NaN literal: the IEEE-754 bit patterns go through
# `bitcast` from the unsigned integer of the same width. Literals without a suffix
# are abstract and work in any typed context, u32 gains the `u` suffix, f16 the `h`.

# ═══════════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════════

# Shared-walk mixin targets, declared so the overlay procs are module-scope
# symbols when the walk's generics resolve (mixin ignores codeReordering).
proc emitTypeName(ctx: var PrintCtx, t: CirType): string
proc coordRef(ctx: var PrintCtx, e: CirExpr): string
proc coordDot(ctx: var PrintCtx, e: CirExpr): string
proc printConv(ctx: var PrintCtx, e: CirExpr): string
proc printSizeof(ctx: var PrintCtx, e: CirExpr): string
proc getSymbolText(ctx: var PrintCtx, e: CirExpr): string
proc getDotText(ctx: var PrintCtx, e: CirExpr, baseTxt: string): string
proc getCtorValue(ctx: var PrintCtx, t: CirType, kvv: CirKv): string
proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string
proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string
proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool
proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string
proc printAsgn(ctx: var PrintCtx, s: CirStmt): string
proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string
proc emitBinding(ctx: var PrintCtx, p: CirParam, i: int): string
proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string
proc printParamBindings(ctx: var PrintCtx, globalFns: seq[CirProc]): string
proc printPrelude(ctx: var PrintCtx, m: CirModule): string
proc getAtomicCallText(ctx: var PrintCtx, callee: CirExpr, argExprs: seq[CirExpr], args: seq[string]): string
proc printCallArgTexts(ctx: var PrintCtx, callee: CirExpr, args: seq[CirExpr]): seq[string]

proc emitWgslScalar(ctx: var PrintCtx, t: CirType): string =
  ## Returns the WGSL name of a non-array type, "" when the caller
  ## builds the type name.
  ##
  ## 64-bit leaves refuse: their device features exist but the vendored
  ## wgpu-native exposes no WGPUFeatureName to request them.
  case t.kind
  of ctVoid:
    result = getScalarName(TargetVocabs[cgWebGPU].scalarNames, t)
  of ctBool, ctInt, ctUint:
    result = getScalarName(TargetVocabs[cgWebGPU].scalarNames, t)
    if result.len == 0:
      if t.bits == 64:
        raiseAssert("i64/u64 need the shader-int64 device feature, the vendored wgpu-native cannot request it")
      else:
        raiseAssert($t.bits & "-bit integers have no WGSL type, only 32")
  of ctFloat:
    result = getScalarName(TargetVocabs[cgWebGPU].scalarNames, t)
    if result.len == 0:
      if t.bits == 64:
        raiseAssert("f64 needs the shader-f64 device feature, the vendored wgpu-native cannot request it")
      else:
        raiseAssert($t.bits & "-bit floats have no WGSL type")
  of ctBf16:
    raiseAssert("bfloat16 has no WGSL type")
  of ctVector:
    let elem = ctx.emitWgslScalar(t.vElem)
    result = getVectorName(wgslVectorRule, TargetVocabs[cgWebGPU].scalarNames, t)
    if result.len == 0 and elem.len > 0:
      raiseAssert("vector of " & elem & "x" & $t.vN & " has no WGSL type name, the vector needs 2 to 4 lanes")
  of ctStruct:
    result = t.name
  of ctString:
    raiseAssert("string has no WGSL type")
  of ctPtr, ctAptr, ctArray, ctProc, ctSpan:
    result = ""

proc emitTypeName(ctx: var PrintCtx, t: CirType): string =
  case t.kind
  of ctArray:
    let elem = ctx.emitTypeName(t.aElem)
    if t.aN == 0:
      result = "array<" & elem & ">"
    else:
      result = "array<" & elem & ", " & $t.aN & ">"
  else:
    result = ctx.emitWgslScalar(t)

proc structFieldIsBool(t: CirType, field: string): bool =
  if t.isNil or t.kind != ctStruct:
    return false
  for f in t.fields:
    if f.name == field:
      return f.typ.kind == ctBool
  return false

# ═══════════════════════════════════════════════════════════════════════
#  Coordinate builtins
# ═══════════════════════════════════════════════════════════════════════

type WgslBuiltinParam = tuple[attribute, param, wgslType: string]
  ## Canonical coordinate builtin name -> WGSL `@builtin(...)` globalFn parameter.
  ## - `attribute`: the WGSL builtin attribute.
  ## - `param`: the identifier the globalFn body reads.
  ## - `wgslType`: the WGSL type (scalar for the flat thread index).

proc getWgslBuiltinParam(k: CirCoordBuiltinKind): WgslBuiltinParam =
  ## Returns the WGSL parameter of a coordinate builtin kind, an empty tuple when none exists.
  ## `threads_per_threadgroup` is absent, WGSL exposes the threadgroup size
  ## only as the `@workgroup_size` entrypoint attribute, the printer refuses a read.
  case k
  of cbkThreadPositionInGrid:
    ("global_invocation_id", "global_id", "vec3<u32>")
  of cbkThreadgroupPositionInGrid:
    ("workgroup_id", "workgroup_id", "vec3<u32>")
  of cbkThreadPositionInThreadgroup:
    ("local_invocation_id", "local_invocation_id", "vec3<u32>")
  of cbkThreadgroupsPerGrid:
    ("num_workgroups", "num_workgroups", "vec3<u32>")
  of cbkThreadIndexInThreadgroup:
    ("local_invocation_index", "local_invocation_index", "u32")
  else:
    ("", "", "")

proc getWgslCoordComponent(k: CirCoordBuiltinKind, field: string): string =
  case field
  of "x", "y", "z":
    let p = getWgslBuiltinParam(k)
    if p.attribute.len > 0:
      result = "i32(" & p.param & "." & field & ")"
  else:
    result = ""

proc getWgslWholeVector(k: CirCoordBuiltinKind): string =
  let p = getWgslBuiltinParam(k)
  result = "vec3<i32>(i32(" & p.param & ".x), i32(" & p.param & ".y), i32(" &
    p.param & ".z))"

const wgslEntryInputs*: EntryInputs = block:
  ## WGSL entry-input text per coordinate builtin kind: the `@builtin(...)` parameter that the entry signature declares.
  ##
  ## - built from the `getWgslBuiltinParam` table, one text per kind that has a parameter
  ## - a kind without a WGSL builtin has no text: the printer refuses a `threads_per_threadgroup` read,
  ##   WGSL has only the `@workgroup_size` attribute
  var inputs: EntryInputs
  for k in cbkThreadPositionInGrid..cbkThreadIndexInThreadgroup:
    let p = getWgslBuiltinParam(k)
    if p.attribute.len > 0:
      inputs[k] = "@builtin(" & p.attribute & ") " & p.param & ": " & p.wgslType
  inputs

proc coordRef(ctx: var PrintCtx, e: CirExpr): string =
  ## One coordinate builtin read in the globalFn body. WGSL builtins are
  ## entrypoint parameters. A device function read refuses.
  if not ctx.inGlobalFn:
    raiseAssert("coordinate builtin " & $e.coordBuiltin & " in a device function, WGSL builtins are entrypoint parameters")
  if e.coordBuiltin == cbkThreadsPerThreadgroup:
    raiseAssert("threads_per_threadgroup has no WGSL builtin, WGSL has only the @workgroup_size attribute")
  ctx.builtins.incl(e.coordBuiltin)
  if e.coordBuiltin == cbkThreadIndexInThreadgroup:
    return "i32(" & getWgslBuiltinParam(e.coordBuiltin).param & ")"
  getWgslWholeVector(e.coordBuiltin)

proc coordDot(ctx: var PrintCtx, e: CirExpr): string =
  ## One coordinate builtin component read in the globalFn body.
  if not ctx.inGlobalFn:
    raiseAssert("a device function reads " & $e.dotBase.coordBuiltin & ", WGSL coordinate builtins are entrypoint parameters")
  let mapped = getWgslCoordComponent(e.dotBase.coordBuiltin, e.fieldName)
  if mapped.len == 0:
    if e.dotBase.coordBuiltin == cbkThreadsPerThreadgroup:
      raiseAssert("threads_per_threadgroup reads refuse on WGSL, only the @workgroup_size attribute exists")
    else:
      raiseAssert("coordinate builtin has no component " & e.fieldName)
    return ""
  ctx.builtins.incl(e.dotBase.coordBuiltin)
  mapped

# ═══════════════════════════════════════════════════════════════════════
#  Expressions
# ═══════════════════════════════════════════════════════════════════════

proc printConv(ctx: var PrintCtx, e: CirExpr): string =
  ## WGSL has no bool constructor and no bool-to-numeric conversion.
  if e.castType.kind == ctBool:
    raiseAssert("conversion to bool, WGSL has no bool constructor")
  let bt = getExprType(e.castOperand)
  if bt != nil and bt.kind == ctBool:
    raiseAssert("conversion from bool, WGSL has no bool-to-numeric conversion")
  ctx.emitTypeName(e.castType) & "(" & ctx.printExpr(e.castOperand) & ")"

proc printSizeof(ctx: var PrintCtx, e: CirExpr): string =
  raiseAssert(getUnhandledExprRejection(e.kind))

proc bindsNimBool(ctx: var PrintCtx, name: string): bool =
  ## A globalFn bool parameter or a bool workgroup variable stores as the vocab's `storageBool` word,
  ## reads coerce to `bool`, writes wrap in `NIM_BOOL()`.
  for p in getEntryGlobalFn(ctx, ctx.m).params:
    if p.name == name and p.typ.kind == ctBool:
      return true
  for g in ctx.m.globals:
    if g.kind == sVar and g.addressSpace == caThreadgroup and
        g.declType.kind == ctBool and g.name == name:
      return true
  false

proc getSymbolText(ctx: var PrintCtx, e: CirExpr): string =
  if e.typ != nil and e.typ.kind == ctBool and ctx.bindsNimBool(e.name):
    return "bool(" & e.name & ")"
  e.name

proc getDotText(ctx: var PrintCtx, e: CirExpr, baseTxt: string): string =
  if structFieldIsBool(getExprType(e.dotBase), e.fieldName):
    return "bool(" & baseTxt & "." & e.fieldName & ")"
  baseTxt & '.' & e.fieldName

proc getCtorValue(ctx: var PrintCtx, t: CirType, kvv: CirKv): string =
  if structFieldIsBool(t, kvv.name):
    return "NIM_BOOL(" & ctx.printExpr(kvv.value) & ")"
  ctx.printExpr(kvv.value)

proc printExtraExpr(ctx: var PrintCtx, e: CirExpr): string =
  ## WGSL routes the literal, ternary and addressing kinds itself.
  case e.kind
  of eFloatLit:
    # every bitcast<f64> operand form fails, a u32-terminated 64-bit literal is a parse error
    if e.fValue.classify in {fcInf, fcNegInf, fcNan} and e.fLitType.bits == 64:
      raiseAssert("a 64-bit non-finite float has no WGSL token, bitcast<f64> needs the unrequestable shader-f64 feature")
    emitLiteral(e, TargetVocabs[ctx.backend].lit, TargetVocabs[ctx.backend].scalarNames)
  of eTernary:
    # select(f, t, cond) puts the false value first, both branches evaluate
    # eagerly and a faulting read in either branch runs whatever the condition
    "select(" & ctx.printExpr(e.elseE) & ", " & ctx.printExpr(e.thenE) &
      ", " & ctx.printExpr(e.cond) & ")"
  of eAt, ePat:
    let base = if e.kind == eAt: e.atBase else: e.patBase
    var (baseTxt, idxTxt) = ctx.printSubscriptParts(e)
    let bt = getExprType(base)
    if bt != nil and bt.kind in {ctPtr, ctAptr} and
        not (bt.kind == ctAptr and bt.addressSpace == caDevice):
      baseTxt = "(*" & baseTxt & ")[" & idxTxt & "]"
    else:
      baseTxt = baseTxt & "[" & idxTxt & "]"
    baseTxt
  else:
    raiseAssert(getUnhandledExprRejection(e.kind))


proc getStmtBuiltinText(ctx: var PrintCtx, s: CirStmt, pad: string, args: seq[string]): string =
  ## WGSL has no combined barrier.
  case s.callCallee.synchroBuiltin
  of cbkThreadgroupBarrier:
    pad & "workgroupBarrier();"
  of cbkThreadgroupBarrierDevice:
    pad & "workgroupBarrier();\n" & pad & "storageBarrier();"
  else:
    ""

# ═══════════════════════════════════════════════════════════════════════
#  Statement declarations
# ═══════════════════════════════════════════════════════════════════════

proc checkLocalVar(ctx: var PrintCtx, s: CirStmt): bool =
  ## WGSL locals are function or workgroup storage, the WGSL lowering pass
  ## hoists the storage spaces to program scope.
  if s.addressSpace == caThreadgroup:
    raiseAssert("local " & s.name & " in threadgroup addressSpace, the WGSL lowering pass gives it program scope")
  elif s.addressSpace == caConstant:
    raiseAssert(s.name & " is a constant-addressSpace local the WGSL lowering pass relocates to program scope")
  elif s.addressSpace == caDevice:
    raiseAssert("local " & s.name & " in device addressSpace, WGSL locals are function/workgroup")
  true

proc emitLocalDecl(ctx: var PrintCtx, s: CirStmt): string =
  "var " & s.name & ": " & ctx.emitTypeName(s.declType)

proc printAsgn(ctx: var PrintCtx, s: CirStmt): string =
  ## One assignment. A NIM_BOOL storage target coerces the stored value:
  ## a bool parameter, a bool struct member, a bool workgroup variable.
  let pad = spaces(ctx.indent * 2)
  var boolTarget = ""
  if s.lhs.kind == eSym and ctx.bindsNimBool(s.lhs.name):
    boolTarget = s.lhs.name
  elif s.lhs.kind == eDot and
      structFieldIsBool(getExprType(s.lhs.dotBase), s.lhs.fieldName):
    boolTarget = ctx.printExpr(s.lhs.dotBase) & "." & s.lhs.fieldName
  if boolTarget.len > 0:
    return pad & boolTarget & " = NIM_BOOL(" & ctx.printExpr(s.rhs) & ");"
  result = pad & ctx.printExpr(s.lhs) & " = " & ctx.printExpr(s.rhs) & ";"

# ═══════════════════════════════════════════════════════════════════════
#  Procs and modules
# ═══════════════════════════════════════════════════════════════════════

proc printProcHead(ctx: var PrintCtx, fn: CirProc): string =
  ## Head of one proc, signature and open brace:
  ##
  ## - a device function: the direct `fn name(params) -> ret` signature, a void return emits no arrow (WGSL has no `void` type name)
  ## - a pointer parameter takes the `ptr<function, T>` form
  ## - the entry globalFn: `@compute @workgroup_size(...)` attribute lines, then `fn <entry>(<entry inputs>)` over the body-walk inputs
  ##   exportc renames the entry for the host runtime lookup.
  if fn.isGlobalFn:
    let wg = getWorkgroup(fn, [64, 1, 1])
    let entryName = if fn.exportName.len > 0: fn.exportName else: fn.name
    result = "@compute @workgroup_size(" & $wg[0] & ", " & $wg[1] & ", " &
      $wg[2] & ")\nfn " & entryName & "(" &
      emitEntryInputs(ctx.builtins, wgslEntryInputs) & ") {\n"
    return
  var params: seq[string]
  for p in fn.params:
    case p.typ.kind
    of ctPtr, ctAptr:
      # without `wgslPointerParams` only the ptr<function> form, the call inlining removes the rest
      var elem = ctx.emitTypeName(p.typ.pElem)
      if (fn.name & "\x1f" & p.name) in ctx.atomics:
        elem = "atomic<" & elem & ">"
      if p.typ.kind == ctAptr:
        elem = "array<" & elem & ">"
      let space = case p.typ.addressSpace
        of caDevice: "storage"
        of caThreadgroup: "workgroup"
        else: "function"
      if space != "function" and not wgslPointerParams:
        raiseAssert("parameter " & p.name & " of " & fn.name & " points into " & space & ", WGSL needs -d:wgslPointerParams for it")
      params.add p.name & ": ptr<" & space & ", " & elem &
        (if space == "storage": ", read_write>" else: ">")
    else:
      params.add p.name & ": " & ctx.emitTypeName(p.typ)
  result = "fn " & fn.name & "(" & params.join(", ") & ")"
  if fn.ret.kind != ctVoid:
    result &= " -> " & ctx.emitTypeName(fn.ret)

proc emitComputeGlobal(ctx: var PrintCtx, s: CirStmt): string =
  ## Program-scope declaration text of one relocated storage local.
  ## No initializer survives relocation, the rewrite ran before printing.
  if s.addressSpace == caThreadgroup:
    let ty = if s.declType.kind == ctBool:
      TargetVocabs[cgWebGPU].storageBool
    else:
      ctx.emitTypeName(s.declType)
    return "var<workgroup> " & s.name & ": " & ty & ";"
  if s.init == nil:
    ctx.refuseUninitializedConstMem(s.name)
    return ""
  printDeclStmt("", "const " & s.name & ": " & ctx.emitTypeName(s.declType), ctx.printExpr(s.init))

proc emitBinding(ctx: var PrintCtx, p: CirParam, i: int): string =
  ## `@group(0) @binding(i)` declaration of one globalFn parameter in host `store` packing order:
  ## - device buffers bind writable storage, `var<storage, read_write>`
  ## - scalars and value structs bind read-only storage, packed in the engine scratch buffer
  let attr = "@group(0) @binding(" & $i & ") "
  case p.typ.kind
  of ctAptr:
    var elem = ctx.emitTypeName(p.typ.pElem)
    if (getEntryGlobalFn(ctx, ctx.m).name & "\x1f" & p.name) in ctx.atomics:
      elem = "atomic<" & elem & ">"
    result = attr & "var<storage, read_write> " & p.name & ": array<" & elem & ">;"
  of ctPtr:
    raiseAssert("globalFn parameter " & p.name & " of pointer type has no WGSL binding")
  of ctArray:
    raiseAssert("globalFn parameter " & p.name & " of array type has no WGSL binding")
  of ctStruct:
    for f in p.typ.fields:
      if f.typ.kind == ctVector:
        raiseAssert("globalFn parameter " & p.name & " field " & f.name & " is a vector with no host-packed WGSL storage")
      if f.typ.kind == ctArray and f.typ.aElem.kind == ctBool:
        raiseAssert("globalFn parameter " & p.name & " field " & f.name & " is a bool array, only a scalar bool field maps to a NIM_BOOL word")
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitTypeName(p.typ) & ";"
  of ctBool:
    result = attr & "var<storage, read> " & p.name & ": " &
      TargetVocabs[cgWebGPU].storageBool & ";"
  else:
    result = attr & "var<storage, read> " & p.name & ": " &
      ctx.emitTypeName(p.typ) & ";"

proc emitStructField(ctx: var PrintCtx, t: CirType, f: CirField): string =
  ## Dissolution passes rewrite every use of a pointer-holding member before printing.
  if f.typ.containsPtrDeep:
    return ""
  result = f.name & ": " & (if f.typ.kind == ctBool:
    TargetVocabs[cgWebGPU].storageBool
  else:
    ctx.emitTypeName(f.typ))

proc printParamBindings(ctx: var PrintCtx, globalFns: seq[CirProc]): string =
  for i, p in globalFns[0].params:
    result &= ctx.emitBinding(p, i) & "\n"
  if result.len > 0:
    result &= "\n"

proc printPrelude(ctx: var PrintCtx, m: CirModule): string =
  ## `enable f16` from the stored module feature scan `CirModule.features`,
  ## filled by the feature-scan pass, then the `NIM_BOOL` alias when interface
  ## storage or a declared struct field is bool.
  var usesNimBool = false
  block scan:
    for p in getEntryGlobalFn(ctx, m).params:
      if p.typ.kind == ctBool:
        usesNimBool = true
        break scan
    for g in m.globals:
      if g.kind == sVar and g.addressSpace == caThreadgroup and g.declType.kind == ctBool:
        usesNimBool = true
        break scan
    for t in m.types:
      if t.kind == ctStruct:
        for f in t.fields:
          if f.typ.kind == ctBool:
            usesNimBool = true
            break scan
  if m.features.fp16:
    result = declareExtensions(m.features, [(true, "enable f16;\n")])
  when wgslPointerParams:
    block pointers:
      for fn in m.procs:
        if not fn.isGlobalFn:
          for p in fn.params:
            if p.typ.kind in {ctPtr, ctAptr} and p.typ.addressSpace != caThread:
              result &= "requires unrestricted_pointer_parameters, pointer_composite_access;\n\n"
              break pointers
  if usesNimBool:
    result &= "alias NIM_BOOL = u32;\n\n"
  when wgslPreciseMath:
    result &= wgslPreciseMathSource & "\n"

proc rootSymbol(e: CirExpr): string =
  ## The variable an access chain starts from, "" for none.
  var x = e
  while x != nil:
    case x.kind
    of eAddr: x = x.addrOperand
    of eAt: x = x.atBase
    of ePat: x = x.patBase
    of eDeref: x = x.derefOperand
    of eCast, eConv: x = x.castOperand
    of eSym: return x.name
    else: return ""
  ""

proc scanAtomics(m: CirModule): HashSet[string] =
  ## `proc\x1fsymbol` of every buffer an atomic builtin reaches, through the
  ## pointer parameters that pass it on: WGSL types them `atomic<T>`.
  var procs: Table[string, CirProc]
  for fn in m.procs:
    procs[fn.name] = fn
  proc calls(fn: CirProc, visit: proc(callee: CirExpr, args: seq[CirExpr])) =
    fn.body.walkStmts(proc(s: CirStmt) =
      if s.kind == sCall: visit(s.callCallee, s.callArgs),
    proc(e: CirExpr) =
      if e.kind == eCall: visit(e.callee, e.args))
  var direct: seq[string]
  for fn in m.procs:
    let name = fn.name
    fn.calls(proc(callee: CirExpr, args: seq[CirExpr]) =
      if callee.kind == eSym and callee.atomicsBuiltin != cbkNone and args.len > 0:
        let r = rootSymbol(args[0])
        if r.len > 0: direct.add(name & "\x1f" & r))
  for d in direct:
    result.incl d
  var changed = true
  while changed:
    changed = false
    for fn in m.procs:
      let name = fn.name
      var found: seq[string]
      let marked = result
      fn.calls(proc(callee: CirExpr, args: seq[CirExpr]) =
        if callee.kind != eSym: return
        let g = procs.getOrDefault(callee.name)
        if g == nil: return
        for j, p in g.params:
          if j < args.len and (g.name & "\x1f" & p.name) in marked:
            let r = rootSymbol(args[j])
            if r.len > 0 and (name & "\x1f" & r) notin marked:
              found.add(name & "\x1f" & r))
      for f in found:
        result.incl f
        changed = true

proc getAtomicCallText(ctx: var PrintCtx, callee: CirExpr, argExprs: seq[CirExpr], args: seq[string]): string =
  ## `atomicAdd` over an `atomic<T>` element, the buffer's type from `scanAtomics`.
  "atomicAdd(" & args[0] & ", " & args[1] & ")"

proc printCallArgTexts(ctx: var PrintCtx, callee: CirExpr, args: seq[CirExpr]): seq[string] =
  ## A device-buffer binding passes on as the address of its array.
  let entry = getEntryGlobalFn(ctx, ctx.m)
  for a in args:
    var txt = ctx.printExpr(a)
    if wgslPointerParams and ctx.inGlobalFn and a.kind == eSym and a.typ != nil and a.typ.kind == ctAptr and
        a.typ.addressSpace == caDevice:
      for p in entry.params:
        if p.name == a.name:
          txt = "(&" & txt & ")"
    result.add txt

const wgslPreciseMath* {.booldefine.} = false
  ## acos, asin, sin, cos and log as accurate f32 functions (to 4e-7) in
  ## place of WGSL's builtins: acos and asin inherit atan2's 4096 ULP, sin
  ## and cos are good to 2^-11 absolute (and measured 2e-4 on SwiftShader)

const wgslPreciseMathSource = """
// accurate f32 math for WGSL (whose acos/asin inherit atan2's 4096 ULP,
// sin/cos 2^-11 absolute): Abramowitz-Stegun 4.4.46, Cephes sinf/cosf/logf
fn precise_acos(x: f32) -> f32 {
  let a = min(abs(x), 1.0);
  var p = -0.0012624911;
  p = p * a + 0.0066700901; p = p * a - 0.0170881256; p = p * a + 0.0308918810;
  p = p * a - 0.0501743046; p = p * a + 0.0889789874; p = p * a - 0.2145988016;
  p = p * a + 1.5707963050;
  let r = sqrt(1.0 - a) * p;
  return select(r, 3.14159265358979 - r, x < 0.0);
}
fn precise_asin(x: f32) -> f32 { return 1.57079632679490 - precise_acos(x); }
fn precise_sincos_poly(y: f32, q: i32) -> vec2<f32> {
  let z = y * y;
  let s = y + y * z * ((-1.9515295891e-4 * z + 8.3321608736e-3) * z - 1.6666654611e-1);
  let c = 1.0 - 0.5 * z + z * z * ((2.443315711809948e-5 * z - 1.388731625493765e-3) * z + 4.166664568298827e-2);
  // (sin, cos) of y + q pi/2
  switch (q & 3) {
    case 0: { return vec2<f32>(s, c); }
    case 1: { return vec2<f32>(c, -s); }
    case 2: { return vec2<f32>(-s, -c); }
    default: { return vec2<f32>(-c, s); }
  }
}
fn precise_reduce(x: f32) -> vec2<f32> {
  let j = round(x * 0.636619772367581);
  let y = ((x - j * 1.5703125) - j * 4.8375129699707031e-4) - j * 7.5497899548918821e-8;
  return vec2<f32>(y, j);
}
fn precise_sin(x: f32) -> f32 { let r = precise_reduce(x); return precise_sincos_poly(r.x, i32(r.y)).x; }
fn precise_cos(x: f32) -> f32 { let r = precise_reduce(x); return precise_sincos_poly(r.x, i32(r.y)).y; }
fn precise_log(x0: f32) -> f32 {
  let fr = frexp(x0);
  var x = fr.fract;
  var e = f32(fr.exp);
  if (x < 0.707106781186547) { e = e - 1.0; x = x + x - 1.0; } else { x = x - 1.0; }
  let z = x * x;
  var y = 7.0376836292e-2;
  y = y * x - 1.1514610310e-1; y = y * x + 1.1676998740e-1; y = y * x - 1.2420140846e-1;
  y = y * x + 1.4249322787e-1; y = y * x - 1.6668057665e-1; y = y * x + 2.0000714765e-1;
  y = y * x - 2.4999993993e-1; y = y * x + 3.3333331174e-1;
  y = y * x * z;
  y = y - 2.12194440e-4 * e;
  y = y - 0.5 * z;
  return x + y + 0.693359375 * e;
}
"""

const wgslPreciseNames = [("acos", "precise_acos"), ("acosf", "precise_acos"),
  ("asin", "precise_asin"), ("asinf", "precise_asin"), ("sin", "precise_sin"),
  ("sinf", "precise_sin"), ("cos", "precise_cos"), ("cosf", "precise_cos"),
  ("log", "precise_log"), ("logf", "precise_log"), ("ln", "precise_log")]

proc printWgslModule*(m: CirModule): string =
  var ctx = initPrintCtx(wgslWalkPolicy, wgslVectorRule, wgslEmit, cgWebGPU)
  ctx.mathNames = wgslMathBuiltinNames
  when wgslPreciseMath:
    for (c, precise) in wgslPreciseNames:
      ctx.mathNames[(c, 32)] = precise
  let globalFn = getEntryGlobalFn(ctx, m)
  ctx.m = m
  ctx.atomics = scanAtomics(m)
  printModule(ctx, m, @[globalFn], wgslModuleShape)

proc codegen*(m: CirModule): string =
  codegenModule(printWgslModule(m))
