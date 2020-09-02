option (ENABLE_QUNIT_CPU_PARALLEL "Make QEngineCPU partially async, so QUnit can parallelize over it (may exceed shell threading limits)" OFF)

if (ENABLE_QUNIT_CPU_PARALLEL)
    set(QUNIT_CPU_PARALLEL ON)
endif (ENABLE_QUNIT_CPU_PARALLEL)
