//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "qfactory.hpp"

#define CATCH_CONFIG_RUNNER /* Access to the configuration. */
#include "tests.hpp"

using namespace Qrack;

enum QInterfaceEngine testEngineType = QINTERFACE_CPU;
enum QInterfaceEngine testSubEngineType = QINTERFACE_CPU;
enum QInterfaceEngine testSubSubEngineType = QINTERFACE_CPU;
qrack_rand_gen_ptr rng;
bool enable_normalization = false;
bool disable_hardware_rng = false;
bool async_time = false;
bool sparse = false;
int device_id = -1;
bitLenInt max_qubits = 24;
bool single_qubit_run = false;
std::string mOutputFileName;
std::ofstream mOutputFile;
bool isBinaryOutput = false;

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool qengine = false;
    bool qunit = false;
    bool cpu = false;
    bool opencl_single = false;
    bool opencl_multi = false;

    int mxQbts = 24;

    using namespace Catch::clara;

    /*
     * Allow specific layers and processor types to be enabled.
     */
    auto cli = session.cli() | Opt(qengine)["--layer-qengine"]("Enable Basic QEngine tests") |
        Opt(qunit)["--layer-qunit"]("Enable QUnit implementation tests") |
        Opt(cpu)["--proc-cpu"]("Enable the CPU-based implementation tests") |
        Opt(opencl_single)["--proc-opencl-single"]("Single (parallel) processor OpenCL tests") |
        Opt(opencl_multi)["--proc-opencl-multi"]("Multiple processor OpenCL tests") |
        Opt(async_time)["--async-time"]("Time based on asynchronous return") |
        Opt(enable_normalization)["--enable-normalization"](
            "Enable state vector normalization. (Usually not "
            "necessary, though might benefit accuracy at very high circuit depth.)") |
        Opt(disable_hardware_rng)["--disable-hardware-rng"]("Modern Intel chips provide an instruction for hardware "
                                                            "random number generation, which this option turns off. "
                                                            "(Hardware generation is on by default, if available.)") |
        Opt(device_id, "device-id")["-d"]["--device-id"]("Opencl device ID (\"-1\" for default device)") |
        Opt(mxQbts, "max-qubits")["-m"]["--max-qubits"](
            "Maximum qubits for test (default value 24, enter \"-1\" for automatic selection)") |
        Opt(mOutputFileName, "measure-output")["--measure-output"](
            "Specifies a file name for bit measurement outputs. If specificed, benchmark iterations will always be "
            "concluded with a full measurement and written to the given file name, as human-readable or raw integral "
            "binary depending on --binary-output") |
        Opt(isBinaryOutput)["--binary-output"]("If included, specifies that the --measure-output file "
                                               "type should be binary. (By default, it is "
                                               "human-readable.)") |
        Opt(single_qubit_run)["--single"]("Only run single (maximum) qubit count for tests") |
        Opt(sparse)["--sparse"](
            "(For QEngineCPU, under QUnit:) Use a state vector optimized for sparse representation and iteration.");

    session.cli(cli);

    /* Set some defaults for convenience. */
    session.configData().useColour = Catch::UseColour::No;
    session.configData().rngSeed = std::time(0);

    // session.configData().abortAfter = 1;

    /* Parse the command line. */
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }

    session.config().stream() << "Random Seed: " << session.configData().rngSeed;

    if (disable_hardware_rng) {
        session.config().stream() << std::endl;
    } else {
        session.config().stream() << " (Overridden by hardware generation!)" << std::endl;
    }

    if (!qengine && !qunit) {
        qunit = true;
        qengine = true;
    }

    if (!cpu && !opencl_single && !opencl_multi) {
        cpu = true;
        opencl_single = true;
        opencl_multi = true;
    }

    if (mxQbts == -1) {
        // If we're talking about a particular OpenCL device,
        // we have an API designed to tell us device capabilities and limitations,
        // like maximum RAM allocation.
        if (opencl_single) {
#if ENABLE_OPENCL
            // Make sure the context singleton is initialized.
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset();

            DeviceContextPtr device_context = OCLEngine::Instance()->GetDeviceContextPtr(device_id);
            size_t maxMem = device_context->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / sizeof(complex);
            size_t maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / sizeof(complex);

            // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
            max_qubits = log2(maxAlloc);
            if ((QEngineOCL::OclMemDenom * pow2(max_qubits)) > maxMem) {
                max_qubits = log2(maxMem / QEngineOCL::OclMemDenom);
            }
#else
            // With OpenCL tests disabled, it's ambiguous what device we want to set the limit by.
            // If we're not talking about the OpenCL resources of a single device,
            // maximum allocation becomes a notoriously thorny matter.
            // For any case besides the above, we just use the default.
#endif
        }
    } else {
        max_qubits = mxQbts;
    }

    if (mOutputFileName.compare("")) {
        session.config().stream() << "Measurement results output file: " << mOutputFileName << std::endl;
        if (isBinaryOutput) {
            mOutputFile.open(mOutputFileName, std::ios::out | std::ios::binary);
        } else {
            mOutputFile.open(mOutputFileName, std::ios::out);
            mOutputFile << "TestName, QubitCount, MeasurementResult" << std::endl;
        }
    }

    int num_failed = 0;

    if (num_failed == 0 && qengine) {
        /* Perform the run against the default (software) variant. */
        if (num_failed == 0 && cpu) {
            testEngineType = QINTERFACE_CPU;
            testSubEngineType = QINTERFACE_CPU;
            testSubSubEngineType = QINTERFACE_CPU;
            session.config().stream() << "############ QEngine -> CPU ############" << std::endl;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QEngine -> OpenCL ############" << std::endl;
            testEngineType = QINTERFACE_OPENCL;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qunit) {
        testEngineType = QINTERFACE_QUNIT;
        if (num_failed == 0 && cpu) {
            if (sparse) {
                session.config().stream() << "############ QUnit -> QEngine -> CPU (Sparse) ############" << std::endl;
            } else {
                session.config().stream() << "############ QUnit -> QEngine -> CPU ############" << std::endl;
            }
            testSubEngineType = QINTERFACE_CPU;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QUnit -> QEngine -> OpenCL ############" << std::endl;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && opencl_multi) {
            session.config().stream() << "############ QUnitMulti (OpenCL) ############" << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif
    }

    if (mOutputFileName.compare("")) {
        mOutputFile.close();
    }

    return num_failed;
}

QInterfaceTestFixture::QInterfaceTestFixture()
{
    uint32_t rngSeed = Catch::getCurrentContext().getConfig()->rngSeed();

    if (rngSeed == 0) {
        rngSeed = std::time(0);
    }

    qrack_rand_gen_ptr rng = std::make_shared<qrack_rand_gen>();
    rng->seed(rngSeed);

    qftReg = NULL;
}
