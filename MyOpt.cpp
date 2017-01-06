#include "MyOpt.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cstdio>
#include "def.h"
using namespace std;
using namespace Eigen;
#define TODO                                    \
    {                                           \
        cerr << "TODO assertion" << endl;       \
        cerr << "\tFile: " << __FILE__ << endl; \
        cerr << "\tLine: " << __LINE__ << endl; \
        cerr << "\tFunc: " << __func__ << endl; \
        assert(false);                          \
    }
MyOpt::MyOpt(MyOpt::Algorithm a, size_t dim)
    : _algo(a), _dim(dim), _cond(_default_stop_cond()), _data(nullptr), _solver(nullptr)
{
}
StopCond MyOpt::_default_stop_cond()
{
    StopCond sc;
    sc.stop_val = -1 * INF;
    sc.xtol_rel = 1e-15;
    sc.ftol_rel = 1e-15;
    sc.gtol     = 1e-15;
    sc.history  = 2;
    sc.max_iter = 100;
    sc.max_eval = 300;
    return sc;
}
void MyOpt::set_min_objective(ObjFunc f, void* d)
{
    _func = f;
    _data = d;
}
MyOpt::Result MyOpt::optimize(VectorXd& x0, double& y)
{
    switch (_algo)
    {
        case CG:
            _solver = new class CG(_func, _dim, _cond, _data);
            break;
        case BFGS:
            _solver = new class BFGS(_func, _dim, _cond, _data);
            break;
        case RProp:
            _solver = new class RProp(_func, _dim, _cond, _data);
            break;
        default:
            cerr << "Unsupported algorithm, tag: " << _algo << endl;
            return INVALID_ARGS;
    }
    _solver->set_param(_params);
    return _solver->minimize(x0, y);
}
MyOpt::~MyOpt()
{
    if (_solver != nullptr) delete _solver;
}
void MyOpt::set_stop_val(double v) { _cond.stop_val = v; }
void MyOpt::set_xtol_rel(double v) { _cond.xtol_rel = v; }
void MyOpt::set_ftol_rel(double v) { _cond.ftol_rel = v; }
void MyOpt::set_gtol(double v)     { _cond.gtol     = v; }
void MyOpt::set_history(size_t h)  { _cond.history  = h + 1; }
void MyOpt::set_max_eval(size_t v) { _cond.max_eval = v; }
void MyOpt::set_max_iter(size_t v) { _cond.max_iter = v; }
void MyOpt::set_algo_param(const std::map<std::string, double>& p) { _params = p; }
std::string MyOpt::get_algorithm_name() const noexcept
{
#define C(A) case A: return #A
    switch (_algo)
    {
        C(CG);
        C(BFGS);
        C(RProp);
        default:
            return "Unsupported algorithm";
    }
#undef C
}
std::string MyOpt::explain_result(Result r) const noexcept
{
#define C(A) case A: return #A
    switch(r)
    {
        C(FAILURE         );
        C(INVALID_ARGS    );
        C(INVALID_INITIAL );
        C(NANINF          );
        C(SUCCESS         );
        C(STOPVAL_REACHED);
        C(FTOL_REACHED);
        C(XTOL_REACHED);
        C(GTOL_REACHED);
        C(MAXEVAL_REACHED);
        C(MAXITER_REACHED);
        default:
            return "Unknown reason" + to_string(r);
    }
#undef C
}
size_t MyOpt::get_dimension() const noexcept { return _dim; }
Solver::Solver(ObjFunc f, size_t dim, StopCond sc, void* d)
    : _func(f),
      _line_search_trial(1.0),
      _eval_counter(0),
      _iter_counter(0),
      _dim(dim),
      _cond(sc),
      _data(d),
      _result(MyOpt::SUCCESS), 
      _history_x(queue<VectorXd>()),
      _history_y(queue<double>()), 
      _c_decrease(0.01), 
      _c_curvature(0.9)
{}
double Solver::_run_func(const VectorXd& x, VectorXd& g, bool need_g)
{
    const double val = _func(x, g, need_g, _data);
    // automatically record best evaluation and update _eval_counter
    ++_eval_counter;
    if (val < _besty)
    {
        _bestx = x; 
        _besty = val;
    }
    return val;
}
Solver::~Solver() {}
void Solver::_init()
{
    _eval_counter = 0;
    _iter_counter = 0;
    while (!_history_x.empty())
    {
        _history_x.pop();
        _history_y.pop();
    }
    _bestx = VectorXd::Constant(_dim, 1, INF);
    _besty = INF;
    _current_x = VectorXd::Constant(_dim, 1, INF);
    _current_g = VectorXd::Constant(_dim, 1, INF);
    _current_y = INF;
}
MyOpt::Result Solver::minimize(VectorXd& x0, double& y)
{
    _init();
    _current_x = x0;
    _current_g = VectorXd::Constant(_dim, 1, INF);
    _current_y = _run_func(_current_x, _current_g, true);
#ifdef MYDEBUG
    printf("Initial y = %g, G.norm = %g\n\n", _current_y, _current_g.norm());
#endif

    while (!_limit_reached())
    {
        _one_iter();
        ++_iter_counter;
        _update_hist();
    }
    x0 = _bestx;
    y  = _besty;
    return _result;
}
void Solver::_update_hist()
{
    assert(_history_x.size() == _history_y.size());
    _history_x.push(_current_x);
    _history_y.push(_besty);
    while (_history_x.size() > _cond.history)
    {
        _history_x.pop();
        _history_y.pop();
    }
}
void Solver::set_param(const map<string, double>& p) { _params = p; }
bool Solver::_limit_reached()
{
    if (_result == MyOpt::SUCCESS)
    {
        if (_besty < _cond.stop_val)
            _result = MyOpt::STOPVAL_REACHED;
        else if (_history_x.size() >= _cond.history &&
                 (_history_x.front() - _history_x.back()).norm() < _cond.xtol_rel * (1 + _history_x.front().norm()))
            _result = MyOpt::XTOL_REACHED;
        else if (_history_y.size() >= _cond.history &&
                 fabs(_history_y.front() - _history_y.back()) < _cond.ftol_rel * (1 + fabs(_history_y.front())))
            _result = MyOpt::FTOL_REACHED;
        else if (_current_g.norm() < _cond.gtol)
            _result = MyOpt::GTOL_REACHED;
        else if (_eval_counter > _cond.max_eval)
            _result = MyOpt::MAXEVAL_REACHED;
        else if (_iter_counter > _cond.max_iter)
            _result = MyOpt::MAXITER_REACHED;
    }
    return _result != MyOpt::SUCCESS;
}
void Solver::_set_linesearch_factor(double c1, double c2)
{
    assert(0 <= c1 && c1 <= c2 && c2 <= 1.0);
    _c_decrease  = c1;
    _c_curvature = c2;
}
void Solver::_set_trial(double t) { _line_search_trial = t; }
double Solver::_get_trial() const noexcept { return _line_search_trial; }
bool Solver::_line_search_inexact(const Eigen::VectorXd& direction, double& alpha, Eigen::VectorXd& x,
                                  Eigen::VectorXd& g, double& y, size_t max_search)
{
    // Basically translated from minimize.m of gpml toolbox
    const double trial = _get_trial();
    const size_t init_eval = _eval_counter;
    const double interpo   = 0.618;
    const double extrapo   = 1 / 0.618;
    const double rho       = _c_decrease; // sufficient decrease
    const double sig       = _c_curvature;  // curvature
    const double d0        = _current_g.dot(direction);
    const double f0        = _current_y;
    double x2, f2, d2; VectorXd df2(_dim);
    double x3, f3, d3; VectorXd df3(_dim);
    assert(d0 <= 0);
    // extrapolation
    x3 = trial; f3 = INF; d3 = INF;
    while(true)
    {
        x2 = 0;     f2 = f0;  d2 = d0;
        while(_eval_counter - init_eval < max_search)
        {
            f3 = _run_func(_current_x + x3 * direction, df3, true);
            d3 = df3.dot(direction);
#ifdef MYDEBUG
            printf("Extrapolation, x3 = %g, f3 = %g\n", x3, f3);
#endif
            if(std::isnan(f3) || std::isnan(d3) || std::isinf(f3) || std::isinf(d3))
                x3 /= 2;
            else
                break;
        }
        if(d3 > sig * d0 || f3 > f0 + x3 * rho * d0)
            break;

        double x1, f1, d1;
        x1 = x2; f1 = f2; d1 = d2;
        x2 = x3; f2 = f3; d2 = d3;
        const double A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 // make cubic extrapolation
        const double B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
        x3 = x1 - d1 * pow(x2 - x1, 2) / (B + sqrt(B * B - A * d1 * (x2 - x1)));  // num. error possible, ok!

        if (std::isnan(x3) || std::isinf(x3) || x3 < 0)  // num prob | wrong sign?
            x3 = x2 * extrapo;                           // extrapolate maximum amount
        else if (x3 > x2 * extrapo)                      // new point beyond extrapolation limit?
            x3 = x2 * extrapo;                           // extrapolate maximum amount
        else if (x3 < x2 + interpo * (x2 - x1))          // new point too close to previous point?
            x3 = x2 + interpo * (x2 - x1);
    }

    // Interpolation
    double x4 = INF, f4 = -1*INF, d4 = INF;
    while ((abs(d3) > -1*sig*d0 || f3 > f0 + x3 * rho * d0) && _eval_counter - init_eval < max_search)
    {
        if (d3 > 0 || f3 > f0+x3*rho*d0)                         // choose subinterval
        {
          x4 = x3; f4 = f3; d4 = d3;                      // move point 3 to point 4
        }
        else
        {
          x2 = x3; f2 = f3; d2 = d3;                      // move point 3 to point 2
        }

        if (f4 > f0)           
            x3 = x2 - (0.5 * d2 * pow(x4 - x2, 2)) / (f4 - f2 - d2 * (x4 - x2));  // quadratic interpolation
        else
        {
            const double A  = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2);  // cubic interpolation
            const double B  = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2);
            x3 = x2 + (sqrt(B * B - A * d2 * pow(x4 - x2, 2)) - B) / A;  // num. error possible, ok!
        }
        if (isnan(x3) || isinf(x3))
            x3 = (x2+x4)/2;               // if we had a numerical problem then bisect
        x3 = max(min(x3, x4-interpo*(x4-x2)),x2+interpo*(x4-x2));  // don't accept too close
        f3 = _run_func(_current_x + x3 * direction, df3, true);
#ifdef MYDEBUG
        printf("Interpolation, x3 = %g, f3 = %g\n", x3, f3);
#endif
        d3 = df3.dot(direction);
    }

    alpha = x3;
    x     = _current_x + alpha * direction;
    g     = df3;
    y     = f3;
    if(! (abs(d3) < -sig*d0 && f3 < f0+x3*rho*d0))
    {
#ifdef MYDEBUG
        cerr << "Wolfe condition violated" << endl;
#endif
        return false;
    }
    return true;
}
void CG::_init() 
{ 
    Solver::_init(); 
    _former_g         = VectorXd(_dim);
    _former_direction = VectorXd(_dim);
    _set_linesearch_factor(0.05, 0.1);
}
double CG::_beta_FR() const noexcept
{
    return _current_g.squaredNorm() / _former_g.squaredNorm();
};
double CG::_beta_PR() const noexcept
{
    return std::max(0.0, _current_g.dot(_current_g - _former_g) / _former_g.squaredNorm());
};
MyOpt::Result CG::_one_iter() 
{
    const size_t inner_iter = _iter_counter % _dim;
    double trial     = 1.0 / (1 + _current_g.squaredNorm());
    if(_iter_counter == 0)
        _set_trial(trial);

    VectorXd direction(_dim);
    if(inner_iter == 0)
        direction = -1 * _current_g;
    else
    {
        // double beta = _beta_FR();
        double beta = _beta_PR();
        direction = -1 * _current_g + beta * _former_direction;
    }
    double alpha       = 0;
    double y           = 0;
    VectorXd x(_dim);
    VectorXd g(_dim);
    _line_search_inexact(direction, alpha, x, g, y, 20);
    const double ratio = 10;
    if(inner_iter != 0)
        trial = alpha * min(ratio, direction.dot(_current_g) / _former_direction.dot(_former_g));
    else
        trial = alpha;
    _set_trial(trial);
    _former_g         = _current_g;
    _former_direction = direction;
    _current_x        = x;
    _current_g        = g;
    _current_y        = y;
#ifdef MYDEBUG
    printf("Iter = %zu, Eval = %zu, InnerIter = %zu, Y = %g, G.norm = %g, alpha = %g, new trial = %g\n", _iter_counter, _eval_counter, inner_iter, _current_y, _current_g.norm(), alpha, trial);
    cout << "=======================" << endl;
#endif
    return MyOpt::SUCCESS;
}
void BFGS::_init() 
{ 
    Solver::_init(); 
    _set_linesearch_factor(1e-4, 0.9); // suggested by <Numerical Optimization>
    _invB = MatrixXd::Identity(_dim, _dim);
}
MyOpt::Result BFGS::_one_iter() 
{ 
    double trial     = 1.0 / (1 + _current_g.norm());
    if(_iter_counter == 0)
        _set_trial(trial);
    VectorXd direction = -1 * _invB * _current_g;
    double alpha, y;
    VectorXd x(_dim);
    VectorXd g(_dim);
    bool ls_success = _line_search_inexact(direction, alpha, x, g, y, 20); // always use 1.0 as trial
    trial = alpha * 1.1;
    _set_trial(trial);
    if(ls_success)
    {
        VectorXd sk = alpha * direction;
        VectorXd yk = g - _current_g;
        _invB = _invB + ((sk.dot(yk) + yk.transpose() * _invB * yk) * (sk * sk.transpose())) / pow(sk.dot(yk), 2)
            - (_invB * yk * sk.transpose() + sk * yk.transpose() * _invB) / sk.dot(yk);
    }
    else
    {
#ifdef MYDEBUG
        cout << "Restart" << endl;
#endif
        _invB = MatrixXd::Identity(_dim, _dim);
    }
    _current_x = x;
    _current_g = g;
    _current_y = y;
#ifdef MYDEBUG
    printf("Iter = %zu, Eval = %zu, Y = %g, G.norm = %g, alpha = %g\n", _iter_counter, _eval_counter, _current_y,
           _current_g.norm(), alpha);
#endif
    return MyOpt::SUCCESS;
}
void RProp::_init() 
{ 
    Solver::_init(); 
    _delta    = VectorXd::Constant(_dim, 1, _delta0);
    _grad_old = VectorXd::Ones(_dim, 1);
}
MyOpt::Result RProp::_one_iter() 
{ 
    VectorXd sign(_dim);
    for(size_t i = 0; i < _dim; ++i)
    {
        sign(i) = _current_g(i) == 0 ? 0 : (_current_g(i) > 0 ? 1 : -1);
        const double changed = _current_g(i) * _grad_old(i);
        if(changed > 0)
            _delta(i) = min(_delta(i) * _eta_plus, _delta_max);
        else if(changed < 0)
        {
            _delta(i)     = max(_delta(i) * _eta_minus, _delta_min);
            _current_g(i) = 0;
            sign(i)       = 0;
        }
    }
    VectorXd this_delta = -1*sign.cwiseProduct(_delta);

    _grad_old      = _current_g;
    VectorXd x_old = _current_x;
    _current_x += this_delta;
    _current_y = _run_func(_current_x, _current_g, true);

    // Recover from inf or NaN
    while (std::isinf(_current_y) || std::isnan(_current_y) || std::isinf(_current_g.squaredNorm()) ||
           std::isnan(_current_g.squaredNorm()))
    {
        this_delta = 0.618 * this_delta;
        _current_x = x_old + this_delta;
        _current_y = _run_func(_current_x, _current_g, true);
#ifdef MYDEBUG
        cout << "Recover, y = " << _current_y << endl;
#endif
    }
#ifdef MYDEBUG
    printf("Iter = %zu, Eval = %zu, Y = %g, G.norm = %g, delta.norm = %g\n", _iter_counter, _eval_counter, _current_y,
           _current_g.norm(), this_delta.norm());
    cout << "=======================" << endl;
#endif
    return MyOpt::SUCCESS;
}
