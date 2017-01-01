#include "MyOpt.h"
#include <cassert>
#include <iostream>
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
    : _eval_counter(0),
      _iter_counter(0),
      _dim(dim),
      _cond(sc),
      _data(d),
      _history_x(queue<VectorXd>()),
      _history_y(queue<double>()),
      _result(MyOpt::SUCCESS)
{
    _func = [&](const VectorXd& x, VectorXd& grad, bool need_g, void* d) -> double {
        double val = f(x, grad, need_g, d);
        if (val < _besty)
        {
            _bestx = x;
            _besty = val;
        }
        ++_eval_counter;
        return val;
    };
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
    _current_y = _func(_current_x, _current_g, true, _data);

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
void Solver::_line_search_exact(const VectorXd& direction, double& alpha, double& y, int max_search, double trial)
{
    // exact line search use Golden search
    // Start from _current_x, _current_y
    // Assume _current_g is the gradient of _current_x
    assert(_current_g.dot(direction) <= 0);
    auto line_func = [&](const double a, double& g, bool need_g) -> double {
        VectorXd x = _current_x + a * direction;
        VectorXd grad(x.size());
        double y = _func(x, grad, need_g, _data);
        g = direction.dot(grad);  // g is invalid if need_g == false
        return y;
    };
    const size_t init_eval = _eval_counter;
    double prev_alpha = 0;
    double line_g = 0;
    double a1 = 0;    // lower bound
    double a2 = INF;  // upper bound
    alpha = trial;
    y = line_func(alpha, line_g, false);
    if (y >= _current_y)
    {
        while (y > _current_y)
        {
            prev_alpha = alpha;
            alpha = alpha / 5.1;
            y = line_func(alpha, line_g, false);
        }
        a2 = prev_alpha;
    }
    else
    {
        double prev_y = y;
        while (y <= prev_y)
        {
            prev_alpha = alpha;
            alpha = alpha * 2;
            y = line_func(alpha, line_g, false);
        }
        a2 = alpha;
    }
    const int remain_search = max_search - (_eval_counter - init_eval);

    // Golden selection
    const double golden_rate = (sqrt(5) - 1) / 2.0;
    double y1 = _current_y;
    double y2 = y;
    double a3, a4, y3, y4;
    for (int i = remain_search - 1; i > 0; --i)
    {
        const double interv_len = a2 - a1;
        a3 = a2 - golden_rate * interv_len;
        a4 = a1 + golden_rate * interv_len;
        if (a3 == a4)
            break;
        else
        {
            assert(a3 < a4);
            y3 = line_func(a3, line_g, false);
            y4 = line_func(a4, line_g, false);
            if (y3 < y4)
            {
                a2 = a4;
                y2 = y4;
            }
            else
            {
                a1 = a3;
                y1 = y3;
            }
        }
    }
    if (y3 < y4)
    {
        alpha = a3;
        y     = y3;
    }
    else
    {
        alpha = a4;
        y     = y4;
    }
    assert(y < _current_y);
}
void CG::_init() 
{ 
    Solver::_init(); 
    _former_g         = VectorXd(_dim);
    _former_direction = VectorXd(_dim);
}
MyOpt::Result CG::_one_iter() 
{
    const size_t inner_iter = _iter_counter % _dim;
    static double trial     = 1.0 / log2(1 + _current_g.norm());
    cout << "Iter: " << _iter_counter << endl;
    cout << "Eval: " << _eval_counter << endl;
    cout << "Inner iter: " << inner_iter << endl;
    cout << "Y = " << _current_y << endl;
    cout << "G.norm = " << _current_g.norm() << endl;
    VectorXd direction(_dim);
    if(inner_iter == 0)
        direction = -1 * _current_g;
    else
        direction = -1 * _current_g + _current_g.squaredNorm() / _former_g.squaredNorm() * _former_direction;
    double alpha = 0;
    double y     = 0;
    _line_search_exact(direction, alpha, y, 40, trial);
    cout << "Alpha: " << alpha << endl;
    _former_g         = _current_g;
    _former_direction = direction;
    _current_x        = _current_x + alpha * direction;
    _current_y        = _func(_current_x, _current_g, true, _data);
    cout << "=======================" << endl;
    return MyOpt::SUCCESS;
}
void BFGS::_init() { Solver::_init(); }
MyOpt::Result BFGS::_one_iter() { TODO; }
void RProp::_init() { TODO; }
MyOpt::Result RProp::_one_iter() { TODO; }
