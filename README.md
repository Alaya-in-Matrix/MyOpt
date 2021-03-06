# README

An unconstrained optimization library, with an interface like nlopt. 

Three algorithms are to be supported:

- Conjugate gradient
- BFGS
- RProp

Target problem: 1D-100D objective function with gradient infomation

Two reasons I make this library:

- Nlopt's L-BFGS does not perform well in my another project, I want't to see if it could be done better
- Nlopt does not provide conjugate gradient and rprop algorithm, I want to compare them with BFGS

## Dependency

- CMake
- Eigen

## Example

```cpp
#include "MyOpt.h"
#include "def.h"
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <functional>
using namespace std;
using namespace Eigen;
std::uniform_real_distribution<double> distr(-100, 100);
double tb_square(const VectorXd& v, VectorXd& grad, bool need_g, void*)
{
    static double x0     = distr(engine);
    static double y0     = distr(engine);
    const double xfactor = 1024;
    double x  = v[0];
    double y  = v[1];
    if(need_g)
    {
        grad[0] = 2 * xfactor * (x - x0);
        grad[1] = 2 * (y - y0);
    }
    return xfactor * pow(x - x0, 2) + pow(y - y0, 2);
}

int main()
{
    printf("SEED is %zu\n", rand_seed);
    MyOpt opt(MyOpt::CG, 2);
    opt.set_min_objective(tb_square, nullptr);
    cout << "Algorithm: " << opt.get_algorithm_name() << endl;
    cout << "Dimension: " << opt.get_dimension() << endl;
    VectorXd x(opt.get_dimension());
    x << distr(engine), distr(engine);
    double y(std::numeric_limits<double>::infinity());
    opt.set_max_iter(5);
    opt.set_max_eval(200);
    opt.set_history(1);
    opt.set_xtol_rel(1e-3);
    opt.set_gtol(1e-3);
    opt.set_ftol_rel(1e-3);
    auto result = opt.optimize(x, y);
    printf("Best x = (%g, %g), y = %g\n", x[0], x[1], y);
    cout << "result is " << result << endl;
    return EXIT_SUCCESS;
}
```
