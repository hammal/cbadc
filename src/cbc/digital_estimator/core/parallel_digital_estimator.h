#ifndef PARALLEL_DIGITAL_ESTIMATOR
#define PARALLEL_DIGITAL_ESTIMATOR
#include "filter.h"
#include <complex>

namespace CBC
{
    typedef std::complex<double> complex;

    class ParallelDigitalEstimator : public Filter
    {
    private:
        complex *lambda_f, *lambda_b, *ff, *fb, *W;
        complex *inital_mean;

    public:
        ParallelDigitalEstimator(complex *lambda_f, complex *lambda_b, complex *ff, complex *fb, complex *W, int K1, int K2, int M, int N, int L);
        ~ParallelDigitalEstimator();
        void compute_batch();
    };
} // namespace CBC
#endif