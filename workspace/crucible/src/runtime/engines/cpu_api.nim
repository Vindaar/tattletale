# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CPU engine types for the Crucible CPU backend.
##
## `CpuEngine` owns the threadpool_compute worker pool.
##
## The engine is exempt from the HwEngine concept:
## arguments are host values, there is no ingest, no artifact and no DeviceMem.

import std/cpuinfo

const cpuPool* = compileOption("threads") and not defined(emscripten)
  ## the worker pool: with threads, but not under Emscripten (no Linux futexes)

when cpuPool:
  import workspace/threadpool_compute

# ═══════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════

type
  CpuEngine* = ref object
    ## CPU execution engine. Owns the worker pool.
    deviceName*: string
    when cpuPool:
      tp*: ptr Threadpool

proc newCpuEngine*(numThreads: int = 0): CpuEngine =
  result = CpuEngine(deviceName: "crucible-cpu")
  when cpuPool:
    let n = if numThreads > 0: numThreads else: countProcessors()
    result.tp = threadpool_new(int32(n))

proc destroyCpuEngine*(engine: CpuEngine) =
  when cpuPool:
    doAssert engine.tp != nil,
      "destroyCpuEngine: the engine's pool is already shut down"
    threadpool_shutdown(engine.tp)
    engine.tp = nil
