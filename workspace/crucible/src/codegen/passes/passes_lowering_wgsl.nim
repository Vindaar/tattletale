# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./passes_lowering_cfamily
import ./passes_low_pointers
import ./pass_helpers

# ############################################################
#
#            WGSL lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerWgslExplodePointerLocals*(reg: var PassRegistry) =
  registerExplodePointerLocals(reg, "wgslExplodePointerLocals",
    "Rewrites pointer locals to their pointees and pointer-holding struct" &
    " locals to their leaf symbols",
    "WGSL")

func registerWgslHoistProgramScopeLocals*(reg: var PassRegistry) =
  registerCfamilyHoistProgramScopeLocals(reg, "wgslHoistProgramScopeLocals",
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope")

const wgslPointerParams* {.booldefine.} = false
  ## Device functions take pointers into storage and workgroup memory
  ## (WGSL `requires unrestricted_pointer_parameters, pointer_composite_access`,
  ## Chrome and Dawn have both): large structs pass by reference as in C
  ## instead of as copies, and no call needs inlining. Storage pointers are
  ## `read_write`, so they point into the device-buffer bindings only.

func registerWgslLowerByrefParams*(reg: var PassRegistry) =
  reg.register("wgslLowerByrefParams", pkTransform,
    "Rewrites passByRef params into pointer params, a call site passes the" &
    " address of its argument or of a function-local copy",
    run = wgslLowerByrefParams)

func registerWgslLowerCompositeTernaries*(reg: var PassRegistry) =
  reg.register("wgslLowerCompositeTernaries", pkTransform,
    "Rewrites a ternary over structs or arrays, which WGSL `select` refuses," &
    " in assignment, declaration or return position into an if-else",
    run = lowerCompositeTernaries)

func registerWgslSplitAliasedPointerArgs*(reg: var PassRegistry) =
  reg.register("wgslSplitAliasedPointerArgs", pkTransform,
    "Copies in and out pointer arguments that share a root variable with an" &
    " earlier argument of the call, WGSL refuses aliased pointer arguments",
    run = splitAliasedPointerArgs)

func registerWgslPruneUnreachable*(reg: var PassRegistry) =
  reg.register("wgslPruneUnreachable", pkTransform,
    "Drops the functions the entry point does not reach (the address-space" &
    " instantiation leaves the base callees behind)",
    run = pruneUnreachable)

func registerWgslInlinePointerParamCalls*(reg: var PassRegistry) =
  reg.register("wgslInlinePointerParamCalls", pkTransform,
    "Inlines calls whose pointer arguments cannot cross a WGSL function" &
    " signature",
    run = inlinePointerParamCalls)

# ═══════════════════════════════════════════════════════════════════════
#  Pointer-param call inlining
# ═══════════════════════════════════════════════════════════════════════

proc crossesWgslSignature(t: CirType): bool =
  ## WGSL function signatures accept values and ptr<function> pointers only,
  ## and storage and workgroup pointers with `wgslPointerParams`.
  if t.isNil:
    return false
  when wgslPointerParams:
    if t.kind in {ctPtr, ctAptr} and t.addressSpace in {caThread, caDevice, caThreadgroup}:
      return false
  case t.kind
  of ctAptr:
    result = true
  of ctPtr:
    result = t.addressSpace != caThread
  else:
    result = false

proc hasCrossingPointerParam(fn: CirProc): bool =
  for p in fn.params:
    if p.typ.crossesWgslSignature:
      return true

proc containsReturn(ss: seq[CirStmt]): bool =
  ## A frontend trailing void-body `return` does not count here: the caller drops it before splicing.
  for s in ss:
    case s.kind
    of sRet:
      return true
    of sIte:
      if s.thenBody.containsReturn or s.elseBody.containsReturn:
        return true
    of sWhile:
      if s.whileBody.containsReturn:
        return true
    else:
      discard
  return false

proc callsCrossingCalleeTransitively(seen: var HashSet[string], fn: CirProc, procs: Table[string, CirProc]): bool =
  ## Returns true when `fn` calls itself or reaches a proc whose parameters
  ## cannot cross a WGSL function signature. Splicing such a cycle would never terminate.
  if fn.name in seen:
    return false
  seen.incl(fn.name)
  var targets: HashSet[string]
  collectCallTargets(targets, fn.body)
  for t in targets:
    let callee = procs.getOrDefault(t)
    if callee.isNil:
      continue
    if callee.hasCrossingPointerParam:
      return true
    if callsCrossingCalleeTransitively(seen, callee, procs):
      return true
  return false

proc substituteArgsExpr(e: CirExpr, subst: Table[string, CirExpr]) =
  if e.isNil:
    return
  if e.kind == eSym:
    if subst.hasKey(e.name):
      let arg = subst[e.name]
      e[] = arg[]
    return
  for ch in getExprChildren(e):
    ch.substituteArgsExpr(subst)

proc substituteArgsStmts(ss: var seq[CirStmt], subst: Table[string, CirExpr]) =
  ## Statement-walk overload of `substituteArgsExpr`.
  ## Hand-rolled walk: a substitution splices the call argument into the symbol node
  ## and never re-walks the spliced content, its symbols belong to the caller.
  ## A shared walk would descend into the rewritten node and substitute a second parameter inside the argument.
  for s in mitems(ss):
    case s.kind
    of sVar:
      s.init.substituteArgsExpr(subst)
    of sAsgn:
      s.lhs.substituteArgsExpr(subst)
      s.rhs.substituteArgsExpr(subst)
    of sIte:
      s.cond.substituteArgsExpr(subst)
      s.thenBody.substituteArgsStmts(subst)
      s.elseBody.substituteArgsStmts(subst)
    of sWhile:
      s.whileCond.substituteArgsExpr(subst)
      s.whileBody.substituteArgsStmts(subst)
    of sRet:
      s.retVal.substituteArgsExpr(subst)
    of sDiscard:
      s.discardVal.substituteArgsExpr(subst)
    of sCall:
      s.callCallee.substituteArgsExpr(subst)
      for a in mitems(s.callArgs):
        a.substituteArgsExpr(subst)
    of sAsm, sEmit:
      discard

proc inlineInStmts(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, procs: Table[string, CirProc])
  ## Forward declaration, nested statement lists recurse into each other.

proc refuseValuePositionCalls(fn: CirProc, e: CirExpr, procs: Table[string, CirProc]) =
  ## A call in value position cannot inline.
  ## Only a void callee splices into statement position, a call whose result
  ## feeds an expression has no destination for the inlined body's writes,
  ## so the pass refuses the call.
  if e.isNil:
    return
  if e.kind == eCall and e.callee.kind == eSym:
    let callee = procs.getOrDefault(e.callee.name)
    if callee != nil and not callee.isGlobalFn and callee.hasCrossingPointerParam:
      raiseAssert("value-position call of " & callee.name & ", its pointer parameter cannot cross a WGSL signature")
  for ch in getExprChildren(e):
    refuseValuePositionCalls(fn, ch, procs)

proc inlineCallSite(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, i: int, procs: Table[string, CirProc]): bool =
  ## Returns whether the statement spliced, refusing calls the splice cannot host.
  let s = ss[i]
  let callee = procs.getOrDefault(s.callCallee.name)
  if callee != nil and not callee.isGlobalFn and callee.hasCrossingPointerParam:
    if callee.ret.kind != ctVoid:
      raiseAssert("inlining " & callee.name & " needs a void return, its pointer parameter cannot cross a WGSL signature")
    var body = cloneStmts(callee.body)
    if body.len > 0 and body[^1].kind == sRet and body[^1].retVal == nil:
      # an implicit fall-through return of the frontend, dropping it is
      # semantics-preserving at the splice position
      body.setLen(body.len - 1)
    if body.containsReturn:
      raiseAssert("inlining " & callee.name & " needs a return-free body, its pointer parameter cannot cross a WGSL signature")
    var seen: HashSet[string]
    if callsCrossingCalleeTransitively(seen, callee, procs):
      raiseAssert("inlining " & callee.name & " reaches a crossing call cycle, its pointer parameter cannot cross a WGSL signature")
    # substitution onto fresh aliases evaluates a call argument once
    var bindings: seq[CirStmt]
    var subst = initTable[string, CirExpr]()
    for j, p in callee.params:
      if p.typ.kind in {ctPtr, ctAptr}:
        # a pointer argument has no WGSL local form, it substitutes directly
        subst[p.name] = s.callArgs[j]
        continue
      var alias = "inl" & $fresh & "_" & p.name
      fresh.inc
      while alias in used:
        alias.add "_"
      used.incl(alias)
      bindings.add CirStmt(kind: sVar, name: alias, declType: p.typ, init: s.callArgs[j])
      subst[p.name] = sym(alias, p.typ)
    body.substituteArgsStmts(subst)
    ss = ss[0 ..< i] & bindings & body & ss[i + 1 .. ^1]
    return true
  for a in s.callArgs:
    refuseValuePositionCalls(fn, a, procs)
  return false

proc walkValuePositions(used: var HashSet[string], fresh: var int, fn: CirProc, s: CirStmt, procs: Table[string, CirProc]) =
  ## One non-call statement: its subexpressions refuse value-position calls,
  ## nested statement lists recurse through `inlineInStmts`.
  case s.kind
  of sVar:
    refuseValuePositionCalls(fn, s.init, procs)
  of sAsgn:
    refuseValuePositionCalls(fn, s.lhs, procs)
    refuseValuePositionCalls(fn, s.rhs, procs)
  of sIte:
    refuseValuePositionCalls(fn, s.cond, procs)
    inlineInStmts(s.thenBody, used, fresh, fn, procs)
    inlineInStmts(s.elseBody, used, fresh, fn, procs)
  of sWhile:
    refuseValuePositionCalls(fn, s.whileCond, procs)
    inlineInStmts(s.whileBody, used, fresh, fn, procs)
  of sRet:
    refuseValuePositionCalls(fn, s.retVal, procs)
  of sDiscard:
    refuseValuePositionCalls(fn, s.discardVal, procs)
  of sCall, sAsm, sEmit:
    discard

proc inlineInStmts(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, procs: Table[string, CirProc]) =
  var i = 0
  while i < ss.len:
    let s = ss[i]
    if s.kind == sCall and s.callCallee.kind == eSym:
      let spliced = inlineCallSite(ss, used, fresh, fn, i, procs)
      if not spliced:
        i.inc
      continue
    walkValuePositions(used, fresh, fn, s, procs)
    i.inc

proc inlinePointerParamCalls(m: CirModule) =
  ## Placement contract:
  ## - the pass runs after the pointer-local explosion pass, so the spliced
  ##   bodies carry no pointer locals
  ## - the per-address-space callee instantiation runs first, a re-spaced formal
  ##   refuses exactly when the new signature cannot print
  var procs = initTable[string, CirProc]()
  for fn in m.procs:
    procs[fn.name] = fn
  for fn in mitems(m.procs):
    var used = initHashSet[string]()
    var locals: seq[tuple[name: string, typ: CirType]]
    collectDeclaredLocals(locals, fn.body)
    for l in locals:
      used.incl(l.name)
    for p in fn.params:
      used.incl(p.name)
    var fresh = 0
    while true:
      inlineInStmts(fn.body, used, fresh, fn, procs)
      var changed = false
      var targets: HashSet[string]
      collectCallTargets(targets, fn.body)
      for t in targets:
        let callee = procs.getOrDefault(t)
        if callee != nil and not callee.isGlobalFn and callee.hasCrossingPointerParam:
          changed = true
      if not changed:
        break
  var kept: seq[CirProc]
  for fn in m.procs:
    if not fn.isGlobalFn and fn.hasCrossingPointerParam:
      # every call site inlined, the signature has no WGSL form
      continue
    kept.add fn
  m.procs = kept


# ═══════════════════════════════════════════════════════════════════════
#  By-ref parameters (wgslPointerParams)
# ═══════════════════════════════════════════════════════════════════════

type ByrefCtx = object
  byref: Table[string, seq[int]]
    ## callee name -> indices of its by-ref formals
  readOnly: HashSet[string]
    ## entry parameters bound as read-only storage values: no `read_write` pointer names them
  globals: Table[string, AddressSpace]
    ## program-scope variables
  tmpCount: int

proc byrefArgSpace(ctx: ByrefCtx, e: CirExpr): tuple[space: AddressSpace, ok: bool] =
  ## Space of the storage an lvalue argument names; `ok` false when no
  ## printable pointer names it (a read-only binding, a constant).
  let (space, found) = pointerSpaceOf(e)
  if found:
    return (space, space in {caThread, caDevice, caThreadgroup})
  var root = e
  while root.kind in {eAt, ePat, eDot}:
    root = case root.kind
      of eAt: root.atBase
      of ePat: root.patBase
      else: root.dotBase
  if root.kind != eSym:
    return (caThread, false)
  if root.name in ctx.readOnly:
    return (caDevice, false)
  if ctx.globals.hasKey(root.name):
    let g = ctx.globals[root.name]
    return (g, g == caThreadgroup)
  (caThread, true)

proc byrefWrapArgs(ctx: var ByrefCtx, pre: var seq[CirStmt], e: CirExpr) =
  ## Nested calls first, so a call sees its arguments already rewritten.
  if e.isNil:
    return
  for ch in getExprChildren(e):
    ctx.byrefWrapArgs(pre, ch)
  if e.kind != eCall or e.callee.kind != eSym:
    return
  for i in ctx.byref.getOrDefault(e.callee.name):
    if i >= e.args.len:
      break
    let a = e.args[i]
    let at = getExprType(a)
    if at == nil or at.kind notin {ctStruct, ctArray}:
      continue
    if a.kind == eDeref and a.derefOperand.kind == eSym:
      # a by-ref formal passed on: the pointer itself, re-spaced with its owner
      e.args[i] = a.derefOperand
      continue
    let (space, ok) = ctx.byrefArgSpace(a)
    if ok and isLvalue(a):
      e.args[i] = addrE(tPtr(space, at), a)
    else:
      inc ctx.tmpCount
      let tmpName = "byref_tmp" & $ctx.tmpCount
      let tmp = sym(tmpName, at)
      pre.add(varD(tmpName, at))
      pre.add(asgnS(tmp, a))
      e.args[i] = addrE(tPtr(caThread, at), tmp)

proc byrefWrapStmts(ctx: var ByrefCtx, stmts: var seq[CirStmt]) =
  var i = 0
  while i < stmts.len:
    var pre: seq[CirStmt] = @[]
    let s = stmts[i]
    case s.kind
    of sAsgn:
      ctx.byrefWrapArgs(pre, s.lhs)
      ctx.byrefWrapArgs(pre, s.rhs)
    of sIte:
      ctx.byrefWrapArgs(pre, s.cond)
      ctx.byrefWrapStmts(s.thenBody)
      ctx.byrefWrapStmts(s.elseBody)
    of sWhile:
      ctx.byrefWrapArgs(pre, s.whileCond)
      ctx.byrefWrapStmts(s.whileBody)
    of sRet:
      ctx.byrefWrapArgs(pre, s.retVal)
    of sDiscard:
      ctx.byrefWrapArgs(pre, s.discardVal)
    of sCall:
      let callE = CirExpr(kind: eCall)
      callE.callee = s.callCallee
      callE.args = s.callArgs
      ctx.byrefWrapArgs(pre, callE)
      s.callCallee = callE.callee
      s.callArgs = callE.args
    of sVar:
      ctx.byrefWrapArgs(pre, s.init)
    of sAsm, sEmit:
      discard
    for h in pre:
      stmts.insert(h, i)
      i.inc
    i.inc

proc wgslLowerByrefParams(m: CirModule) =
  ## Contract:
  ## - a `passByRef` formal (or a large array) of a device function becomes
  ##   a `ptr<function>` to it, its reads deref it; the address-space instantiation
  ##   re-spaces it after the arguments
  ## - a call site passes the argument's address, a by-ref formal passes on
  ##   as itself, a value or read-only binding copies into a fresh local first
  ## - entry functions keep their bindings
  when wgslPointerParams:
    var ctx: ByrefCtx
    for g in m.globals:
      if g.kind == sVar:
        ctx.globals[g.name] = g.addressSpace
    for fn in m.procs:
      if fn.isGlobalFn:
        for p in fn.params:
          if p.typ.kind notin {ctPtr, ctAptr}:
            ctx.readOnly.incl(p.name)
    for fn in m.procs:
      if fn.isGlobalFn:
        continue
      for i, p in fn.params.mpairs:
        # the frontend marks large structs; a large array (a table per gas) goes by reference too
        if not (p.passByRef or p.typ.isLargeArray):
          continue
        let ptrT = tPtr(caThread, p.typ)
        let name = p.name
        p.name = "_p_" & name
        let ptrName = p.name
        fn.body.walkStmts(nil, proc(e: CirExpr) =
          if e.kind == eSym and e.name == name:
            e[] = derefE(sym(ptrName, ptrT))[])
        p.typ = ptrT
        p.passByRef = false
        ctx.byref.mgetOrPut(fn.name, @[]).add(i)
    for fn in m.procs:
      ctx.byrefWrapStmts(fn.body)

# ═══════════════════════════════════════════════════════════════════════
#  Composite ternaries
# ═══════════════════════════════════════════════════════════════════════

proc isComposite(e: CirExpr): bool =
  let t = getExprType(e)
  t != nil and t.kind in {ctStruct, ctArray}

proc lowerCompositeTernaryStmts(ss: var seq[CirStmt]) =
  ## `x = c ? a : b` -> `if c: x = a else: x = b`, a declaration first
  ## declares, a return returns in both branches. Only the chosen branch
  ## evaluates, as in Nim.
  var i = 0
  while i < ss.len:
    let s = ss[i]
    case s.kind
    of sAsgn:
      if s.rhs != nil and s.rhs.kind == eTernary and s.rhs.isComposite:
        let t = s.rhs
        ss[i] = CirStmt(kind: sIte, cond: t.cond,
          thenBody: @[asgnS(s.lhs, t.thenE)], elseBody: @[asgnS(s.lhs, t.elseE)])
        continue
    of sVar:
      if s.init != nil and s.init.kind == eTernary and s.init.isComposite:
        let t = s.init
        s.init = nil
        let target = sym(s.name, s.declType)
        ss.insert(CirStmt(kind: sIte, cond: t.cond,
          thenBody: @[asgnS(target, t.thenE)], elseBody: @[asgnS(target, t.elseE)]), i + 1)
    of sRet:
      if s.retVal != nil and s.retVal.kind == eTernary and s.retVal.isComposite:
        let t = s.retVal
        ss[i] = CirStmt(kind: sIte, cond: t.cond,
          thenBody: @[CirStmt(kind: sRet, retVal: t.thenE)],
          elseBody: @[CirStmt(kind: sRet, retVal: t.elseE)])
        continue
    of sIte:
      lowerCompositeTernaryStmts(s.thenBody)
      lowerCompositeTernaryStmts(s.elseBody)
    of sWhile:
      lowerCompositeTernaryStmts(s.whileBody)
    else:
      discard
    i.inc

proc lowerCompositeTernaries(m: CirModule) =
  for fn in m.procs:
    lowerCompositeTernaryStmts(fn.body)

# ═══════════════════════════════════════════════════════════════════════
#  Aliased pointer arguments
# ═══════════════════════════════════════════════════════════════════════

proc pointerRoot(e: CirExpr): string =
  ## The variable a pointer argument points into, "" for none.
  var x = e
  while x != nil:
    case x.kind
    of eAddr: x = x.addrOperand
    of eAt: x = x.atBase
    of ePat: x = x.patBase
    of eDot: x = x.dotBase
    of eDeref: x = x.derefOperand
    of eSym: return x.name
    else: return ""
  ""

proc splitAliasedArgs(args: var seq[CirExpr], pre, post: var seq[CirStmt], tmpCount: var int) =
  ## A second `&lvalue` argument into a root already passed points to a
  ## fresh local instead, which copies the lvalue in before the call and
  ## out after it: disjoint fields (the usual case) behave the same.
  var roots: HashSet[string]
  for i in 0 ..< args.len:
    let a = args[i]
    let t = getExprType(a)
    if t == nil or t.kind notin {ctPtr, ctAptr}:
      continue
    let r = pointerRoot(a)
    if r.len == 0:
      continue
    if r notin roots:
      roots.incl r
      continue
    if a.kind != eAddr or not isLvalue(a.addrOperand):
      continue
    let vt = getExprType(a.addrOperand)
    if vt == nil:
      continue
    inc tmpCount
    let name = "alias_tmp" & $tmpCount
    let tmp = sym(name, vt)
    pre.add varD(name, vt, a.addrOperand)
    post.add asgnS(a.addrOperand, tmp)
    args[i] = addrE(tPtr(caThread, vt), tmp)

proc splitAliasedStmts(ss: var seq[CirStmt], tmpCount: var int) =
  var i = 0
  while i < ss.len:
    let s = ss[i]
    var pre, post: seq[CirStmt]
    case s.kind
    of sCall:
      splitAliasedArgs(s.callArgs, pre, post, tmpCount)
    of sAsgn:
      if s.rhs != nil and s.rhs.kind == eCall:
        splitAliasedArgs(s.rhs.args, pre, post, tmpCount)
    of sVar:
      if s.init != nil and s.init.kind == eCall:
        splitAliasedArgs(s.init.args, pre, post, tmpCount)
    of sDiscard:
      if s.discardVal != nil and s.discardVal.kind == eCall:
        splitAliasedArgs(s.discardVal.args, pre, post, tmpCount)
    of sIte:
      splitAliasedStmts(s.thenBody, tmpCount)
      splitAliasedStmts(s.elseBody, tmpCount)
    of sWhile:
      splitAliasedStmts(s.whileBody, tmpCount)
    else:
      discard
    for h in pre:
      ss.insert(h, i)
      i.inc
    for k, h in post:
      ss.insert(h, i + 1 + k)
    i += 1 + post.len

proc splitAliasedPointerArgs(m: CirModule) =
  for fn in m.procs:
    var tmpCount = 0
    splitAliasedStmts(fn.body, tmpCount)

# ═══════════════════════════════════════════════════════════════════════
#  Unreachable functions
# ═══════════════════════════════════════════════════════════════════════

proc pruneUnreachable(m: CirModule) =
  var procs = initTable[string, CirProc]()
  for fn in m.procs:
    procs[fn.name] = fn
  var reached: HashSet[string]
  var todo: seq[CirProc]
  for fn in m.procs:
    if fn.isGlobalFn:
      reached.incl fn.name
      todo.add fn
  while todo.len > 0:
    let fn = todo.pop()
    var targets: HashSet[string]
    collectCallTargets(targets, fn.body)
    for t in targets:
      if t notin reached and procs.hasKey(t):
        reached.incl t
        todo.add procs[t]
  var kept: seq[CirProc]
  for fn in m.procs:
    if fn.name in reached:
      kept.add fn
  m.procs = kept
