#ifndef DIGITAL_ESTIMATOR
#define DIGITAL_ESTIMATOR
#include "filter.h"

namespace CBC
{
    class DigitalEstimator : public Filter
    {
    private:
        double *Af, *Ab, *Bf, *Bb, *W;
        double *mean_vector;

    public:
        DigitalEstimator(double *Af, double *Ab, double *Bf, double *Bb, double *W, int K1, int K2, int M, int N, int L);
        ~DigitalEstimator();
        void compute_batch();
    };
} // namespace CBC
#endif