#include <cpuid.h>
#include <immintrin.h>

#pragma once

namespace RdRandWrapper {

  bool getRdRand(unsigned int* pv);

  class RdRandom
  {
  public:

    bool SupportsRDRAND();

    double Next();
  };
}
