//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#include <map>
#include <mutex>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace Qrack {

class OCLDeviceContext;

typedef std::shared_ptr<OCLDeviceContext> DeviceContextPtr;

class OCLDeviceContext {
public:
    cl::Context context;
    cl::CommandQueue queue;
    std::recursive_mutex mutex;
    cl::Kernel apply2x2;
    cl::Kernel apply2x2norm;
    cl::Kernel cohere;
    cl::Kernel decohereprob;
    cl::Kernel decohereamp;
    cl::Kernel disposeprob;
    cl::Kernel prob;
    cl::Kernel x;
    cl::Kernel swap;
    cl::Kernel rol;
    cl::Kernel ror;
    cl::Kernel inc;
    cl::Kernel dec;
    cl::Kernel incc;
    cl::Kernel decc;
    cl::Kernel indexedLda;
    cl::Kernel indexedAdc;
    cl::Kernel indexedSbc;
    cl::Kernel normalize;
    cl::Kernel updatenorm;
};

/** "Qrack::OCLEngine" manages the single OpenCL context. */
class OCLEngine {
public:
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static OCLEngine* Instance();
    /// If this is the first time instantiating the OpenCL context, you may specify the default platform number and default device number.
    static OCLEngine* Instance(int plat, int dev);
    /// Get a pointer to the OpenCL context
    DeviceContextPtr GetDeviceContextPtr(const int& dev = -1);

    int GetDeviceCount() { return deviceCount; }
    int GetDefaultDeviceID() { return default_device_id; };

private:
    int deviceCount;
    int default_device_id;

    std::vector<DeviceContextPtr> all_device_contexts;
    DeviceContextPtr default_device_context;

    OCLEngine(); // Private so that it can  not be called
    OCLEngine(int plat, int dev); // Private so that it can  not be called
    OCLEngine(OCLEngine const&); // copy constructor is private
    OCLEngine& operator=(OCLEngine const& rhs); // assignment operator is private
    static OCLEngine* m_pInstance;

    void InitOCL(int plat, int dev);

    unsigned long PowerOf2LessThan(unsigned long number);
};

} // namespace Qrack
