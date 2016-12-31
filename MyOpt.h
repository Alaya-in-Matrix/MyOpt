#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
typedef std::function<double(const std::vector<double>& input, std::vector<double>& grad, void* data)> ObjFunc;
class Solver;
struct StopCond
{
    double _stop_val;  // when the fom is below _stop_val, stop optimization
    double _xtol_rel;
    double _ftol_rel;
    size_t _history;  // history to remember, for delta based stop condition (xtol and ftol)
    size_t _max_eval;
    size_t _max_iter;
};
class MyOpt
{
public:
    enum Algorithm
    {
        CG = 0,
        BFGS,
        RProp,
        MBFGS, // Google: superlinear nonconvex for papers
        NUM_ALGORITHMS  // number of algorithm
    };
    enum Result
    {  // copied from
        FAILURE = -1,
        INVALID_ARGS = -2,
        INVALID_INITIAL = -3,  // starting point is NAN|INF
        NANINF = -4,           // for those algorithms that can't recover from inf|nan
        SUCCESS = 1,
        STOPVAL_REACHED = 2,
        FTOL_REACHED = 3,
        XTOL_REACHED = 4,
        MAXEVAL_REACHED = 5,
        MAXTIME_REACHED = 6
    };
    MyOpt(Algorithm, size_t);
    Result optimize(std::vector<double>& x0, double& y);
    ~MyOpt();

    std::string opt_message(const Result);
    void set_stop_val(double);
    void set_algo_param(const std::map<std::string, double>&);
    void set_xtol_rel(double);
    void set_history(double);
    void set_ftol_rel(double);
    void set_max_eval(double);
    void set_max_iter(double);  // max line search
    void set_min_objective(ObjFunc, void* data);
    std::string get_algorithm_name() const noexcept;
    size_t get_dimension() const noexcept;

private:
    StopCond _cond;
    ObjFunc _func;
    void* _data;
    Solver* _solver;
    size_t _dim;
    void _default_stop_cond();
};

// Abstract class for solver
class Solver
{
public:
    Solver(ObjFunc, StopCond, void* data);
    void set_param(const std::map<std::string, double>& param);
    virtual MyOpt::Result minimize();

protected:
    StopCond _cond;
    ObjFunc _func;
    void* _data;
    size_t _dim;
    Eigen::MatrixXd _history_x;
    Eigen::VectorXd _history_y;
    std::map<std::string, double> _params;

    virtual void _init() = 0;
    virtual MyOpt::Result _one_iter() = 0;
    virtual bool _to_stop();
};

class CG : public Solver
{
    double c1, c2;  // param to control line search
    void _init();
    MyOpt::Result _one_iter();
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
