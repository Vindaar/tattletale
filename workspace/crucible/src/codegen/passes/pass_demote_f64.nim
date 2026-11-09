# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/sets
import ./pass_datatypes

# ############################################################
#
#            Demotion of double to single precision
#
# ############################################################

# Every 64-bit float of a module becomes a 32-bit float: the types of
# symbols, fields, parameters, literals and operations alike. For code
# written in double precision that runs in single precision on a GPU:
# WebGPU has no doubles, and gaming GPUs run them at a small fraction of
# the speed of singles. The host must then hand over its data in single
# precision too (structs of 4-byte fields have the same layout on both
# sides).

type Demotion = object
  seen: HashSet[pointer]

proc demote(d: var Demotion, t: CirType) =
  if t.isNil or d.seen.containsOrIncl(cast[pointer](t)): return
  case t.kind
  of ctFloat:
    if t.bits == 64: t.bits = 32
  of ctVector: d.demote(t.vElem)
  of ctPtr, ctAptr: d.demote(t.pElem)
  of ctStruct:
    for f in t.fields: d.demote(f.typ)
  of ctArray: d.demote(t.aElem)
  of ctSpan: d.demote(t.sElem)
  of ctProc:
    for p in t.params: d.demote(p)
    d.demote(t.ret)
  of ctInt, ctUint, ctBf16, ctVoid, ctBool, ctString: discard

proc demote(d: var Demotion, s: CirStmt)

proc demote(d: var Demotion, e: CirExpr) =
  if e.isNil: return
  case e.kind
  of eNil: d.demote(e.nilType)
  of eSym: d.demote(e.typ)
  of eIntLit: d.demote(e.litType)
  of eFloatLit: d.demote(e.fLitType)
  of eStrLit, eTrue, eFalse: discard
  of eDeref: d.demote(e.derefOperand)
  of eAt:
    d.demote(e.atBase)
    d.demote(e.atIndex)
  of ePat:
    d.demote(e.patBase)
    d.demote(e.patIndex)
  of eDot: d.demote(e.dotBase)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    d.demote(e.opType)
    d.demote(e.left)
    d.demote(e.right)
  of eNeg:
    d.demote(e.unType)
    d.demote(e.unOperand)
  of eEq, eNeq, eLt, eLe:
    d.demote(e.cmpType)
    d.demote(e.a)
    d.demote(e.b)
  of eNot: d.demote(e.notOperand)
  of eAnd, eOr:
    d.demote(e.andLeft)
    d.demote(e.andRight)
  of eTernary:
    d.demote(e.ternType)
    d.demote(e.cond)
    d.demote(e.thenE)
    d.demote(e.elseE)
  of eCast, eConv:
    d.demote(e.castType)
    d.demote(e.castOperand)
  of eAddr:
    d.demote(e.addrType)
    d.demote(e.addrOperand)
  of eSizeof: d.demote(e.ofType)
  of eOconstr:
    d.demote(e.ocType)
    for kv in e.kvFields: d.demote(kv.value)
  of eAconstr:
    d.demote(e.acType)
    for x in e.elems: d.demote(x)
  of eCall:
    d.demote(e.callee)
    for a in e.args: d.demote(a)
  of eStmts:
    for s in e.stmts: d.demote(s)
    d.demote(e.valueExpr)

proc demote(d: var Demotion, s: CirStmt) =
  if s.isNil: return
  case s.kind
  of sVar:
    d.demote(s.declType)
    d.demote(s.init)
  of sAsgn:
    d.demote(s.lhs)
    d.demote(s.rhs)
  of sIte:
    d.demote(s.cond)
    for x in s.thenBody: d.demote(x)
    for x in s.elseBody: d.demote(x)
  of sWhile:
    d.demote(s.whileCond)
    for x in s.whileBody: d.demote(x)
  of sRet: d.demote(s.retVal)
  of sDiscard: d.demote(s.discardVal)
  of sCall:
    d.demote(s.callCallee)
    for a in s.callArgs: d.demote(a)
  of sAsm:
    for a in s.asmOperands: d.demote(a)
  of sEmit:
    for a in s.emitOperands: d.demote(a)

proc demoteFloat64*(m: CirModule) =
  ## all 64-bit floats of `m` to 32 bits
  var d: Demotion
  for t in m.types: d.demote(t)
  for g in m.globals: d.demote(g)
  for p in m.procs:
    for x in p.params: d.demote(x.typ)
    d.demote(p.ret)
    for s in p.body: d.demote(s)

func registerDemoteFloat64*(reg: var PassRegistry) =
  reg.register("demoteFloat64", pkTransform,
    "Turns every 64-bit float into a 32-bit float (single-precision kernels)",
    run = demoteFloat64)
