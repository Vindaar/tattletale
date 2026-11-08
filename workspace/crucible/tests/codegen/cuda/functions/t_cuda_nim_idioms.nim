## CUDA: plain Nim idioms of host code that device code takes as is
##
## - std/math functions imported from <math.h> (`sqrt`, `ln`, `cos`, `trunc`)
## - an implicit `converter` call
## - `inc`/`dec` statements, on a local and through a var parameter
## - `len` of a fixed-size array
## - arrays indexed from 1 (`array[1..N, T]`), also nested and through a var parameter
## - a generic proc whose body is empty for one instantiation (`when`)
## - an anonymous tuple constructor returned as a named tuple
##
## Run (requires a CUDA GPU):
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/functions/t_cuda_nim_idioms.nim

import std/[unittest, math]
import workspace/crucible

type
  Meters = distinct float64
  Grid = object
    cells: array[1..4, float64]
    rows: array[1..2, array[1..3, float64]]

converter toFloat(x: Meters): float64 = float64(x)

proc pair(a, b: float64): tuple[xx, yy: float64] =
  let c = a * 2.0
  result = (c * c, b * b)

proc onlyWhenTwo[N: static int](x: var float64) =
  when N == 2:
    x = x + 1.0

proc bump(c: var int32, by: int32) =
  inc c, by
  dec c

proc fill(g: var Grid) =
  for i in 1 .. 4:
    g.cells[i] = float64(10 * i)
  for j in 1 .. 2:
    for i in 1 .. 3:
      g.rows[j][i] = float64(10 * j + i)

const kernel = cuda:
  proc idiomsKernel(o: ptr UncheckedArray[float64]) {.global.} =
    o[0] = sqrt(16.0)
    o[1] = ln(1.0)
    o[2] = cos(0.0)
    o[3] = trunc(2.75)
    let d = Meters(2.5)
    o[4] = d * 2.0            # through the converter
    var c = 3'i32
    inc c
    inc c, 5'i32
    dec c, 2'i32
    bump(c, 10'i32)
    o[5] = float64(c)
    var g: Grid
    fill(g)
    o[6] = float64(g.cells.len)
    o[7] = g.cells[1] + g.cells[4]
    o[8] = g.rows[2][3]
    var x = 1.0
    onlyWhenTwo[1](x)
    onlyWhenTwo[2](x)
    o[9] = x
    let p = pair(3.0, 4.0)
    o[10] = p.xx
    o[11] = p.yy

proc runTest() =
  suite "Nim idioms in device code":
    test "math, converter, inc/dec, array len and bounds, empty bodies, tuples":
      var engine = newEngine(runCuda)
      engine.ingest(kernel)
      var o: array[12, float64]
      let dev = engine.allocDevice(float64, o.len)
      engine.launchKernel<<(1, 1)>>("idiomsKernel", dev, ())
      engine.copyToHostFromDevice(addr o[0], dev, o.len)
      check o[0] == 4.0
      check o[1] == 0.0
      check o[2] == 1.0
      check o[3] == 2.0
      check o[4] == 5.0
      check o[5] == 3.0 + 1 + 5 - 2 + 10 - 1
      check o[6] == 4.0
      check o[7] == 10.0 + 40.0
      check o[8] == 23.0
      check o[9] == 2.0
      check o[10] == 36.0
      check o[11] == 16.0

when isMainModule:
  runTest()
