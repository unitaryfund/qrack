#include "rdrandwrapper.hpp"

namespace RdRandWrapper {

bool getRdRand(unsigned int* pv) {
    const int max_rdrand_tries = 10;
    for (int i = 0; i < max_rdrand_tries; ++i) {
      if (_rdrand32_step(pv)) return true;
    }
    return false;
}

bool RdRandom::SupportsRDRAND()
{
    const unsigned int flag_RDRAND = (1 << 30);

    unsigned int eax, ebx, ecx, edx;
    ecx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    return ((ecx & flag_RDRAND) == flag_RDRAND);
}

double RdRandom::Next() {
  unsigned int v;
  double res = 0;
  double part = 1;
  if (!getRdRand(&v)) {
    throw "Failed to get hardware RNG number.";
  }
  v &= 0x7fffffff;
  for (int i = 0; i < 31; i++) {
      part /= 2;
      if (v & (1U << i)) {
          res += part;
      }
  }
  return res;
}

}
