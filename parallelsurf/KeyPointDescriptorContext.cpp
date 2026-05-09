#include <cmath>

#include "KeyPointDescriptorContext.h"

namespace ParallelSurf
{
    KeyPointDescriptorContext::KeyPointDescriptorContext(
        int iSubRegions, int iVecLen,
        double iOrientation) :
        _subRegions(iSubRegions), _sin(sin(iOrientation)), _cos(cos(iOrientation))
    {
        int aExtSub = _subRegions + 2;
        _cmp = new double **[aExtSub];
        for (int aYIt = 0; aYIt < aExtSub; ++aYIt)
        {
            _cmp[aYIt] = new double *[aExtSub];
            for (int aXIt = 0; aXIt < aExtSub; ++aXIt)
            {
                _cmp[aYIt][aXIt] = new double[iVecLen];
                for (int aVIt = 0; aVIt < iVecLen; ++aVIt)
                {
                    _cmp[aYIt][aXIt][aVIt] = 0;
                }
            }
        }
    }

    KeyPointDescriptorContext::~KeyPointDescriptorContext()
    {
        int aExtSub = _subRegions + 2;
        for (int aYIt = 0; aYIt < aExtSub; ++aYIt)
        {
            for (int aXIt = 0; aXIt < aExtSub; ++aXIt)
            {
                delete[] _cmp[aYIt][aXIt];
            }
            delete[] _cmp[aYIt];
        }
    }
}
