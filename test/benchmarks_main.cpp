//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#define CATCH_CONFIG_RUNNER /* Access to the configuration. */
#include "tests.hpp"

#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

using namespace Qrack;

enum QInterfaceEngine testEngineType = QINTERFACE_CPU;
enum QInterfaceEngine testSubEngineType = QINTERFACE_CPU;
enum QInterfaceEngine testSubSubEngineType = QINTERFACE_CPU;
qrack_rand_gen_ptr rng;
bool enable_normalization = false;
bool use_host_dma = false;
bool disable_hardware_rng = false;
bool async_time = false;
bool sparse = false;
int device_id = -1;
bitLenInt max_qubits = 24;
bool single_qubit_run = false;
std::string mOutputFileName;
std::ofstream mOutputFile;
bool isBinaryOutput = false;
int benchmarkSamples = 100;
int benchmarkDepth = 20;
int benchmarkMaxMagic = -1;
int timeout = -1;
std::vector<int64_t> devList;
bool optimal = false;
bool optimal_single = false;

#define SHOW_OCL_BANNER()                                                                                              \
    if (OCLEngine::Instance().GetDeviceCount()) {                                                                      \
        CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset();                                                       \
    }

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool optimal_cpu = false;

    // Layers
    bool qengine = false;
    bool qpager = false;
    bool qunit = false;
    bool qunit_multi = false;
    bool qunit_qpager = false;
    bool qunit_multi_qpager = false;

    // Engines
    bool cpu = false;
    bool opencl = false;
    bool hybrid = false;
    bool bdt = false;
    bool stabilizer = false;
    bool stabilizer_qpager = false;
    bool stabilizer_bdt = false;

    std::string devListStr;

    int mxQbts = 24;

    using namespace Catch::clara;

    /*
     * Allow specific layers and processor types to be enabled.
     */
    auto cli = session.cli() | Opt(qengine)["--layer-qengine"]("Enable Basic QEngine tests") |
        Opt(optimal)["--optimal"]("Run just default optimal (QUnit or QUnitMulti) layer/engine tests") |
        Opt(optimal_single)["--optimal-single"]("Run just default optimal (QUnit only) layer/engine tests") |
        Opt(optimal_cpu)["--optimal-cpu"]("Run just default (CPU-only) optimal layer/engine tests") |
        Opt(qpager)["--layer-qpager"]("Enable QPager implementation tests") |
        Opt(qunit)["--layer-qunit"]("Enable QUnit implementation tests") |
        Opt(qunit_multi)["--layer-qunit-multi"]("Enable QUnitMulti implementation tests") |
        Opt(qunit_qpager)["--layer-qunit-qpager"]("Enable QUnit with QPager implementation tests") |
        Opt(qunit_multi_qpager)["--layer-qunit-multi-qpager"]("Enable QUnitMulti with QPager implementation tests") |
        Opt(stabilizer_qpager)["--proc-stabilizer-qpager"](
            "Enable QStabilizerHybrid over QPager implementation tests") |
        Opt(stabilizer_bdt)["--proc-stabilizer-bdt"](
            "Enable QStabilizerHybrid over QBinaryDecisionTree implementation tests") |
        Opt(cpu)["--proc-cpu"]("Enable the CPU-based implementation tests") |
        Opt(opencl)["--proc-opencl"]("Single (parallel) processor OpenCL tests") |
        Opt(hybrid)["--proc-hybrid"]("Enable CPU/OpenCL hybrid implementation tests") |
        Opt(bdt)["--proc-bdt"]("Enable binary decision tree implementation tests") |
        Opt(stabilizer)["--proc-stabilizer"]("Enable (hybrid) stabilizer implementation tests") |
        Opt(async_time)["--async-time"]("Time based on asynchronous return") |
        Opt(enable_normalization)["--enable-normalization"](
            "Enable state vector normalization. (Usually not "
            "necessary, though might benefit accuracy at very high circuit depth.)") |
        Opt(use_host_dma)["--use-host-dma"](
            "Allocate state vectors as OpenCL host pointers, in an attempt to use Direct Memory Access. This will "
            "probably be slower, and incompatible with OpenCL virtualization, but it can allow greater state vector "
            "buffer RAM width, potentially including swap disk, depending on OpenCL device DMA capabilities.") |
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
            "(For QEngineCPU, under QUnit:) Use a state vector optimized for sparse representation and iteration.") |
        Opt(benchmarkSamples, "samples")["--samples"]("number of samples to collect (default: 100)") |
        Opt(benchmarkDepth, "depth")["--benchmark-depth"](
            "depth of randomly constructed circuits, when applicable, with 1 round of single qubit and 1 round of "
            "multi-qubit gates being 1 unit of depth (default: 20)") |
        Opt(benchmarkMaxMagic, "magic")["--benchmark-max-magic"](
            "max number of t/tadj gates in semi-Clifford tests (default: [defined per test case])") |
        Opt(timeout, "timeout")["--timeout"](
            "Timeout in milliseconds per sample for test_stabilizer_t_nn and test_stabilizer_t_nn_d (default: none)") |
        Opt(devListStr, "devices")["--devices"](
            "list of devices, for QPager (default is solely default OpenCL device)");

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

#if ENABLE_ENV_VARS
    if (getenv("QRACK_QPAGER_DEVICES")) {
        session.config().stream() << "QRACK_QPAGER_DEVICES: " << std::string(getenv("QRACK_QPAGER_DEVICES"))
                                  << std::endl;
    }
    if (getenv("QRACK_QPAGER_DEVICES_HOST_POINTER")) {
        session.config().stream() << "QRACK_QPAGER_DEVICES_HOST_POINTER: "
                                  << std::string(getenv("QRACK_QPAGER_DEVICES_HOST_POINTER")) << std::endl;
    }
#endif

    if (!qengine && !qpager && !qunit && !qunit_multi && !qunit_qpager && !qunit_multi_qpager) {
        qunit = true;
        qunit_multi = true;
        qengine = true;
        // qpager = true;
        // qunit_qpager = true;
        // qunit_multi_qpager = true;
    }

    if (!cpu && !opencl && !hybrid && !bdt && !stabilizer && !stabilizer_qpager && !stabilizer_bdt) {
        cpu = true;
        opencl = true;
        hybrid = true;
        stabilizer = true;
        // bdt = true;
        // stabilizer_qpager = true;
        // stabilizer_bdt = true;
    }

    if (devListStr.compare("") != 0) {
        std::stringstream devListStr_stream(devListStr);
        // See
        // https://stackoverflow.com/questions/7621727/split-a-string-into-words-by-multiple-delimiters#answer-58164098
        std::regex re("[.]");
        while (devListStr_stream.good()) {
            std::string term;
            getline(devListStr_stream, term, ',');
            // the '-1' is what makes the regex split (-1 := what was not matched)
            std::sregex_token_iterator first{ term.begin(), term.end(), re, -1 }, last;
            std::vector<std::string> tokens{ first, last };
            if (tokens.size() == 1U) {
                devList.push_back(stoi(term));
                continue;
            }
            const unsigned maxI = stoi(tokens[0]);
            std::vector<int> ids(tokens.size() - 1U);
            for (unsigned i = 1U; i < tokens.size(); i++) {
                ids[i - 1U] = stoi(tokens[i]);
            }
            for (unsigned i = 0U; i < maxI; i++) {
                for (unsigned j = 0U; j < ids.size(); j++) {
                    devList.push_back(ids[j]);
                }
            }
        }
    }

    if (mxQbts == -1) {
        // If we're talking about a particular OpenCL device,
        // we have an API designed to tell us device capabilities and limitations,
        // like maximum RAM allocation.
        if (opencl || hybrid || stabilizer || stabilizer_qpager) {
#if ENABLE_OPENCL
            // Make sure the context singleton is initialized.
            SHOW_OCL_BANNER();

            DeviceContextPtr device_context = OCLEngine::Instance().GetDeviceContextPtr(device_id);
            size_t maxMem = device_context->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / sizeof(complex);
            size_t maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / sizeof(complex);

            // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
            max_qubits = Qrack::log2(maxAlloc);
            if ((QEngineOCL::OclMemDenom * pow2(max_qubits)) > maxMem) {
                max_qubits = Qrack::log2(maxMem / QEngineOCL::OclMemDenom);
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

#if ENABLE_OPENCL
    SHOW_OCL_BANNER();
#endif

    int num_failed = 0;

    if (num_failed == 0 && optimal) {
        session.config().stream() << "############ Default Optimal (QUnitMulti/QUnit) ############" << std::endl;
        num_failed = session.run();
        return num_failed;
    }

    if (num_failed == 0 && optimal_single) {
        session.config().stream() << "############ Default Optimal (QUnit) ############" << std::endl;
        num_failed = session.run();
        return num_failed;
    }

    if (num_failed == 0 && optimal_cpu) {
        session.config().stream() << "############ Default Optimal (CPU) ############" << std::endl;
        testEngineType = QINTERFACE_QUNIT;
        testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
        testSubSubEngineType = QINTERFACE_CPU;
        num_failed = session.run();
        return num_failed;
    }

    if (num_failed == 0 && qengine) {
        /* Perform the run against the default (software) variant. */
        if (num_failed == 0 && cpu) {
            testEngineType = QINTERFACE_CPU;
            testSubEngineType = QINTERFACE_CPU;
            session.config().stream() << "############ QEngine -> CPU ############" << std::endl;
            num_failed = session.run();
        }

        if (num_failed == 0 && bdt) {
            testEngineType = QINTERFACE_BDT;
            testSubEngineType = QINTERFACE_CPU;
            session.config().stream() << "############ QBinaryDecisionTree ############" << std::endl;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl) {
            session.config().stream() << "############ QEngine -> OpenCL ############" << std::endl;
            testEngineType = QINTERFACE_OPENCL;
            testSubEngineType = QINTERFACE_OPENCL;
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QStabilizerHybrid -> QHybrid ############" << std::endl;
            testEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubEngineType = QINTERFACE_HYBRID;
            num_failed = session.run();
        }
#else
        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QStabilizerHybrid -> QEngineCPU ############" << std::endl;
            testEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qpager) {
        testEngineType = QINTERFACE_QPAGER;
        if (num_failed == 0 && cpu) {
            session.config().stream() << "############ QPager -> QEngine -> CPU ############" << std::endl;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl) {
            session.config().stream() << "############ QPager -> QEngine -> OpenCL ############" << std::endl;
            testSubEngineType = QINTERFACE_OPENCL;
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid) {
            session.config().stream() << "############ QPager -> QEngine -> Hybrid ############" << std::endl;
            testSubEngineType = QINTERFACE_HYBRID;
            num_failed = session.run();
        }
#endif
    }

#if ENABLE_OPENCL
    if (num_failed == 0 && qengine && stabilizer_qpager) {
        testEngineType = QINTERFACE_STABILIZER_HYBRID;
        testSubEngineType = QINTERFACE_QPAGER;
        testSubSubEngineType = QINTERFACE_OPENCL;
        session.config().stream() << "############ QStabilizerHybrid -> QPager -> QEngineOCL ############" << std::endl;
        num_failed = session.run();
    }
#endif

    if (num_failed == 0 && qunit) {
        testEngineType = QINTERFACE_QUNIT;
        if (num_failed == 0 && cpu) {
            if (sparse) {
                session.config().stream() << "############ QUnit -> QEngine -> CPU (Sparse) ############" << std::endl;
            } else {
                session.config().stream() << "############ QUnit -> QEngine -> CPU ############" << std::endl;
            }
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

        if (num_failed == 0 && bdt) {
            session.config().stream() << "############ QUnit -> QBinaryDecisionTree ############" << std::endl;
            testSubEngineType = QINTERFACE_BDT;
            testSubSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl) {
            session.config().stream() << "############ QUnit -> QEngine -> OpenCL ############" << std::endl;
            testSubEngineType = QINTERFACE_OPENCL;
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid) {
            session.config().stream() << "############ QUnit -> QHybrid ############" << std::endl;
            testSubEngineType = QINTERFACE_HYBRID;
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QUnit -> QStabilizerHybrid -> QHybrid ############" << std::endl;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_HYBRID;
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer_bdt) {
            session.config().stream() << "############ QUnit -> QStabilizerHybrid -> QBinaryDecisionTree ############"
                                      << std::endl;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_BDT;
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer_qpager) {
            session.config().stream() << "############ QUnit -> QStabilizerHybrid -> QPager ############" << std::endl;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_QPAGER;
            num_failed = session.run();
        }
    }

    if (num_failed == 0 && qunit_multi) {
        if (num_failed == 0 && opencl) {
            session.config().stream() << "############ QUnitMulti -> QEngineOCL ############" << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid) {
            session.config().stream() << "############ QUnitMulti -> QHybrid ############" << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_HYBRID;
            testSubSubEngineType = QINTERFACE_HYBRID;
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QUnitMulti -> QStabilizerHybrid -> QHybrid ############"
                                      << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_HYBRID;
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer_bdt) {
            session.config().stream()
                << "############ QUnitMulti -> QStabilizerHybrid -> QBinaryDecisionTree ############" << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_BDT;
            num_failed = session.run();
        }
#else
        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QUnit -> QStabilizerHybrid -> QEngineCPU ############"
                                      << std::endl;
            testEngineType = QINTERFACE_QUNIT;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qunit_qpager) {
        testEngineType = QINTERFACE_QUNIT;
        testSubEngineType = QINTERFACE_QPAGER;
        if (num_failed == 0 && cpu) {
            testSubSubEngineType = QINTERFACE_CPU;
            session.config().stream() << "############ QUnit -> QPager -> CPU ############" << std::endl;
            testSubSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl) {
            session.config().stream() << "############ QUnit -> QPager -> OpenCL ############" << std::endl;
            testSubSubEngineType = QINTERFACE_OPENCL;
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid) {
            session.config().stream() << "############ QUnit -> QPager -> QMaskFusion ############" << std::endl;
            testSubEngineType = QINTERFACE_QPAGER;
            testSubSubEngineType = QINTERFACE_MASK_FUSION;
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer_qpager) {
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_QPAGER;
            session.config().stream() << "########### QUnit -> QStabilizerHybrid -> QPager ###########" << std::endl;
            num_failed = session.run();
        }
    }

    if (num_failed == 0 && qunit_multi_qpager && hybrid) {
        session.config().stream() << "############ QUnitMulti -> QPager -> QMaskFusion ############" << std::endl;
        testEngineType = QINTERFACE_QUNIT_MULTI;
        testSubEngineType = QINTERFACE_QPAGER;
        testSubSubEngineType = QINTERFACE_MASK_FUSION;
        num_failed = session.run();
    }

    if (num_failed == 0 && qunit_multi && stabilizer_qpager) {
        testEngineType = QINTERFACE_QUNIT_MULTI;
        testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
        testSubSubEngineType = QINTERFACE_QPAGER;
        session.config().stream() << "########### QUnitMulti -> QStabilizerHybrid -> QPager ###########" << std::endl;
        num_failed = session.run();
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
