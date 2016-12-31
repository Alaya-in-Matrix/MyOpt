#include "MyOpt.h"
#include "def.h"
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <functional>

using namespace std;
using namespace Eigen;

double tb_square(const vector<double>& v, vector<double>& grad, void*)
{
    double x = v[0];
    if(! grad.empty())
        grad[0] = 2 * x;
    return pow(x, 2);
}

int main()
{
    printf("SEED is %zu\n", rand_seed);
    MyOpt opt(MyOpt::BFGS, 1);
    opt.set_min_objective(tb_square, nullptr);
    vector<double> x{34};
    double y(std::numeric_limits<double>::infinity());
    opt.optimize(x, y);
    printf("Best x = %g, y = %g\n", x[0], y);
    return EXIT_SUCCESS;
}
