#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
#include <queue>
typedef std::function<double(const Eigen::VectorXd& input, Eigen::VectorXd& grad, bool need_g, void* data)> ObjFunc;
class Solver;
struct StopCond
{
    double stop_val;  // when the fom is below _stop_val, stop optimization
    double xtol_rel;
    double ftol_rel;
    double gtol;
    size_t history;  // history to remember, for delta based stop condition (xtol and ftol)
    size_t max_iter;
    size_t max_eval;
};
class MyOpt
{
public:
    enum Algorithm
    {
        CG = 0,
        BFGS,
        RProp,
        // MBFGS, // Google: superlinear nonconvex for papers
        // NUM_ALGORITHMS  // number of algorithm
    };
    enum Result
    {  // copied from
        FAILURE         = -1,
        INVALID_ARGS    = -2,
        INVALID_INITIAL = -3,  // starting point is NAN|INF
        NANINF          = -4,           // for those algorithms that can't recover from inf|nan
        SUCCESS         = 0,
        STOPVAL_REACHED,
        FTOL_REACHED,
        XTOL_REACHED,
        GTOL_REACHED,
        MAXEVAL_REACHED,
        MAXITER_REACHED, 
    };
    MyOpt(Algorithm, size_t);
    Result optimize(Eigen::VectorXd& x0, double& y);
    ~MyOpt();

    std::string opt_message(const Result);
    void set_stop_val(double);
    void set_algo_param(const std::map<std::string, double>&);
    void set_xtol_rel(double);
    void set_ftol_rel(double);
    void set_gtol(double);
    void set_history(size_t);
    void set_max_eval(size_t);
    void set_max_iter(size_t);  // max line search
    void set_min_objective(ObjFunc, void* data);
    size_t get_dimension() const noexcept;
    std::string get_algorithm_name() const noexcept;
    std::string explain_result(Result) const noexcept;

private:
    const Algorithm _algo;
    const size_t    _dim;

    StopCond  _cond;
    void*     _data;
    Solver*   _solver;
    ObjFunc   _func;
    std::map<std::string, double> _params;
    StopCond  _default_stop_cond();
};

// Abstract class for solver
class Solver
{
public:
    Solver(ObjFunc, size_t, StopCond, void* data);
    virtual void set_param(const std::map<std::string, double>& param);
    virtual ~Solver();
    virtual MyOpt::Result minimize(Eigen::VectorXd& x0, double& y);

protected:
    size_t _eval_counter;
    size_t _iter_counter;

    size_t          _dim;
    StopCond        _cond;
    void*           _data;
    std::queue<Eigen::VectorXd> _history_x;
    std::queue<double>          _history_y;
    MyOpt::Result   _result;

    ObjFunc         _func;
    std::map<std::string, double> _params;
    Eigen::VectorXd _bestx;
    double _besty;

    // updated by _one_iter()
    Eigen::VectorXd _current_x;
    Eigen::VectorXd _current_g;
    double _current_y;

    virtual void _init(); // clear counter, best_x, best_y, set params
    virtual void _update_hist();
    virtual void _line_search_inexact(const Eigen::VectorXd& direction, double& alpha, Eigen::VectorXd& x,
                                      Eigen::VectorXd& g, double& y, size_t max_search, double trial);
    virtual bool _limit_reached(); // return SUCCESS if not to stop
    virtual MyOpt::Result _one_iter() = 0;
};

class CG : public Solver
{
    double c1, c2;  // param to control line search
    void _init();
    Eigen::VectorXd _former_g;
    Eigen::VectorXd _former_direction;
    MyOpt::Result _one_iter();
    double _beta_FR() const noexcept; // FLETCHER-REEVES update
    double _beta_PR() const noexcept; // POLAK-RIBIERE update
public:
    using Solver::Solver;
};

class BFGS : public Solver
{
    double c1, c2;  // param to control line search
    void _init();
    MyOpt::Result _one_iter();
public:
    using Solver::Solver;
};

class RProp : public Solver
{
    // Some params, need further reading
    void _init();
    MyOpt::Result _one_iter();
public:
    using Solver::Solver;
};
