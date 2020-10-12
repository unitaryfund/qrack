//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "config.h"

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(__APPLE__)
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.hpp>
#elif defined(_WIN32) || ENABLE_SNUCL
#include <CL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

namespace Qrack {

class OCLDeviceCall;

class OCLDeviceContext;

typedef std::shared_ptr<OCLDeviceContext> DeviceContextPtr;
typedef std::shared_ptr<std::vector<cl::Event>> EventVecPtr;

enum OCLAPI {
    OCL_API_UNKNOWN = 0,
    OCL_API_APPLY2X2,
    OCL_API_APPLY2X2_SINGLE,
    OCL_API_APPLY2X2_NORM_SINGLE,
    OCL_API_APPLY2X2_DOUBLE,
    OCL_API_APPLY2X2_WIDE,
    OCL_API_APPLY2X2_SINGLE_WIDE,
    OCL_API_APPLY2X2_NORM_SINGLE_WIDE,
    OCL_API_APPLY2X2_DOUBLE_WIDE,
    OCL_API_PHASE_SINGLE,
    OCL_API_PHASE_SINGLE_WIDE,
    OCL_API_INVERT_SINGLE,
    OCL_API_INVERT_SINGLE_WIDE,
    OCL_API_UNIFORMLYCONTROLLED,
    OCL_API_COMPOSE,
    OCL_API_COMPOSE_WIDE,
    OCL_API_COMPOSE_MID,
    OCL_API_DECOMPOSEPROB,
    OCL_API_DECOMPOSEAMP,
    OCL_API_DISPOSE,
    OCL_API_PROB,
    OCL_API_PROBREG,
    OCL_API_PROBREGALL,
    OCL_API_PROBMASK,
    OCL_API_PROBMASKALL,
    OCL_API_PROBPARITY,
    OCL_API_X_SINGLE,
    OCL_API_X_SINGLE_WIDE,
    OCL_API_Z_SINGLE,
    OCL_API_Z_SINGLE_WIDE,
    OCL_API_ROL,
    OCL_API_INC,
    OCL_API_CINC,
    OCL_API_INCDECC,
    OCL_API_INCS,
    OCL_API_INCDECSC_1,
    OCL_API_INCDECSC_2,
    OCL_API_INCBCD,
    OCL_API_INCDECBCDC,
    OCL_API_INDEXEDLDA,
    OCL_API_INDEXEDADC,
    OCL_API_INDEXEDSBC,
    OCL_API_HASH,
    OCL_API_APPROXCOMPARE,
    OCL_API_NORMALIZE,
    OCL_API_NORMALIZE_WIDE,
    OCL_API_UPDATENORM,
    OCL_API_APPLYM,
    OCL_API_APPLYMREG,
    OCL_API_PHASEFLIP,
    OCL_API_ZEROPHASEFLIP,
    OCL_API_CPHASEFLIPIFLESS,
    OCL_API_PHASEFLIPIFLESS,
    OCL_API_MUL,
    OCL_API_DIV,
    OCL_API_MULMODN_OUT,
    OCL_API_IMULMODN_OUT,
    OCL_API_POWMODN_OUT,
    OCL_API_CMUL,
    OCL_API_CDIV,
    OCL_API_CMULMODN_OUT,
    OCL_API_CIMULMODN_OUT,
    OCL_API_CPOWMODN_OUT,
    OCL_API_FULLADD,
    OCL_API_IFULLADD,
    OCL_API_CLEARBUFFER
};

struct OCLKernelHandle {
    OCLAPI oclapi;
    std::string kernelname;

    OCLKernelHandle(OCLAPI o, std::string kn)
        : oclapi(o)
        , kernelname(kn)
    {
    }
};

class OCLDeviceCall {
protected:
    std::lock_guard<std::mutex> guard;

public:
    // A cl::Kernel is unique object which should always be taken by reference, or the OCLDeviceContext will lose
    // ownership.
    cl::Kernel& call;
    OCLDeviceCall(const OCLDeviceCall&);

protected:
    OCLDeviceCall(std::mutex& m, cl::Kernel& c)
        : guard(m)
        , call(c)
    {
    }

    friend class OCLDeviceContext;

private:
    OCLDeviceCall& operator=(const OCLDeviceCall&) = delete;
};

class OCLDeviceContext {
public:
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    int context_id;
    int device_id;
    cl::CommandQueue queue;
    EventVecPtr wait_events;

protected:
    std::mutex waitEventsMutex;
    std::map<OCLAPI, cl::Kernel> calls;
    std::map<OCLAPI, std::unique_ptr<std::mutex>> mutexes;

public:
    OCLDeviceContext(cl::Platform& p, cl::Device& d, cl::Context& c, int dev_id, int cntxt_id)
        : platform(p)
        , device(d)
        , context(c)
        , context_id(cntxt_id)
        , device_id(dev_id)
    {
        cl_int error;
        queue = cl::CommandQueue(context, d, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        if (error != CL_SUCCESS) {
            queue = cl::CommandQueue(context, d);
        }

        wait_events =
            std::shared_ptr<std::vector<cl::Event>>(new std::vector<cl::Event>(), [](std::vector<cl::Event>* vec) {
                vec->clear();
                delete vec;
            });
    }

    OCLDeviceCall Reserve(OCLAPI call) { return OCLDeviceCall(*(mutexes[call]), calls[call]); }

    EventVecPtr ResetWaitEvents()
    {
        std::lock_guard<std::mutex> guard(waitEventsMutex);
        EventVecPtr waitVec = std::move(wait_events);
        wait_events =
            std::shared_ptr<std::vector<cl::Event>>(new std::vector<cl::Event>(), [](std::vector<cl::Event>* vec) {
                vec->clear();
                delete vec;
            });
        return waitVec;
    }

    void LockWaitEvents() { waitEventsMutex.lock(); }

    void UnlockWaitEvents() { waitEventsMutex.unlock(); }

    void WaitOnAllEvents()
    {
        std::lock_guard<std::mutex> guard(waitEventsMutex);
        if ((wait_events.get())->size()) {
            cl::Event::waitForEvents((const std::vector<cl::Event>&)*(wait_events.get()));
            wait_events->clear();
        }
    }

    size_t GetPreferredConcurrency()
    {
        size_t nrmGroupSize =
            calls[OCL_API_APPLY2X2_SINGLE].getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
        size_t procElemCount = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        size_t maxWorkItems = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];

        size_t nrmGroupCount = procElemCount * nrmGroupSize;
        if (nrmGroupCount > maxWorkItems) {
            nrmGroupCount = maxWorkItems;
        }

        return nrmGroupCount;
    }

    friend class OCLEngine;
};

/** "Qrack::OCLEngine" manages the single OpenCL context. */
class OCLEngine {
public:
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static OCLEngine* Instance();
    /// Get a pointer one of the available OpenCL contexts, by its index in the list of all contexts.
    DeviceContextPtr GetDeviceContextPtr(const int& dev = -1);
    /// Get the list of all available devices (and their supporting objects).
    std::vector<DeviceContextPtr> GetDeviceContextPtrVector();
    /** Set the list of DeviceContextPtr object available for use. If one takes the result of
     * GetDeviceContextPtrVector(), trims items from it, and sets it with this method, (at initialization, before any
     * QEngine objects depend on them,) all resources associated with the removed items are freed.
     */
    void SetDeviceContextPtrVector(std::vector<DeviceContextPtr> vec, DeviceContextPtr dcp = nullptr);
    /// Get the count of devices in the current list.
    int GetDeviceCount() { return all_device_contexts.size(); }
    /// Get default device ID.
    int GetDefaultDeviceID() { return default_device_context->device_id; }
    /// Pick a default device, for QEngineOCL instances that don't specify a preferred device.
    void SetDefaultDeviceContext(DeviceContextPtr dcp);
    /// Initialize the OCL environment, with the option to save the generated binaries. Binaries will be saved/loaded
    /// from the folder path "home". This returns a Qrack::OCLInitResult object which should be passed to
    /// SetDeviceContextPtrVector().
    static void InitOCL(bool buildFromSource = false, bool saveBinaries = false, std::string home = "*");
    /// Get default location for precompiled binaries:
    static std::string GetDefaultBinaryPath()
    {
        if (getenv("QRACK_OCL_PATH")) {
            std::string toRet = std::string(getenv("QRACK_OCL_PATH"));
            if ((toRet.back() != '/') && (toRet.back() != '\\')) {
#if defined(_WIN32) && !defined(__CYGWIN__)
                toRet += "\\";
#else
                toRet += "/";
#endif
            }
            return toRet;
        }
#if defined(_WIN32) && !defined(__CYGWIN__)
        return std::string(getenv("HOMEDRIVE") ? getenv("HOMEDRIVE") : "") +
            std::string(getenv("HOMEPATH") ? getenv("HOMEPATH") : "") + "\\.qrack\\";
#else
        return std::string(getenv("HOME") ? getenv("HOME") : "") + "/.qrack/";
#endif
    }

private:
    static const std::vector<OCLKernelHandle> kernelHandles;
    static const std::string binary_file_prefix;
    static const std::string binary_file_ext;
    std::vector<DeviceContextPtr> all_device_contexts;
    DeviceContextPtr default_device_context;

    OCLEngine(); // Private so that it can  not be called
    OCLEngine(OCLEngine const&); // copy constructor is private
    OCLEngine& operator=(OCLEngine const& rhs); // assignment operator is private
    static OCLEngine* m_pInstance;

    /// Make the program, from either source or binary
    static cl::Program MakeProgram(bool buildFromSource, cl::Program::Sources sources, std::string path,
        std::shared_ptr<OCLDeviceContext> devCntxt);
    /// Save the program binary:
    static void SaveBinary(cl::Program program, std::string path, std::string fileName);

    unsigned long PowerOf2LessThan(unsigned long number);
};

} // namespace Qrack
