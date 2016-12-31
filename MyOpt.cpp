#include "MyOpt.h"
#include <cassert>
#include <iostream>
using namespace std;
using namespace Eigen;
#define TODO                                  \
    {                                         \
        cerr << "TODO assertion" << endl;     \
        cerr << "File: " << __FILE__ << endl; \
        cerr << "Line: " << __LINE__ << endl; \
        cerr << "Func: " << __func__ << endl; \
        assert(false);                        \
    }
MyOpt::MyOpt(MyOpt::Algorithm, size_t) { TODO; }
MyOpt::Result MyOpt::optimize(vector<double>&, double&) { TODO; }
MyOpt::~MyOpt() { TODO; }
void MyOpt::set_stop_val(double) { TODO; }
void MyOpt::set_algo_param(const std::map<std::string, double>&) { TODO; }
void MyOpt::set_xtol_rel(double) { TODO; }
void MyOpt::set_history(double) { TODO; }
void MyOpt::set_ftol_rel(double) { TODO; }
void MyOpt::set_max_eval(double) { TODO; }
void MyOpt::set_max_iter(double) { TODO; }  // max line search
void MyOpt::set_min_objective(ObjFunc, void*) { TODO; }
std::string MyOpt::get_algorithm_name() const noexcept { TODO; }
size_t MyOpt::get_dimension() const noexcept { TODO; }
void MyOpt::_default_stop_cond() { TODO; }
void Solver::set_param(const map<string, double>&) { TODO; }
MyOpt::Result Solver::minimize() { TODO; }
bool Solver::_to_stop() { TODO; }
void CG::_init() { TODO; }
MyOpt::Result CG::_one_iter() { TODO; }
void BFGS::_init() { TODO; }
MyOpt::Result BFGS::_one_iter() { TODO; }
void RProp::_init() { TODO; }
MyOpt::Result RProp::_one_iter() { TODO; }
