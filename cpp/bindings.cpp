#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <fatrop/fatrop.hpp>
#include <fatrop/ip_algorithm/ip_data.hpp>
#include <fatrop/ip_algorithm/ip_timings.hpp>
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace fatrop;
using Array = nb::ndarray<nb::numpy, const Scalar, nb::c_contig, nb::device::cpu>;
using Info = ProblemInfo<OcpType>;

namespace {

// Own the copy, including empty arrays; no callback ever sees solver storage.
nb::object snapshot(const Scalar *data, Index size) {
    auto buffer = std::make_unique<Scalar[]>(std::max<Index>(size, 1));
    if (size) std::copy_n(data, size, buffer.get());
    nb::capsule owner(buffer.get(), [](void *p) noexcept { delete[] static_cast<Scalar *>(p); });
    Scalar *ptr = buffer.release();
    auto result = nb::ndarray<nb::numpy, Scalar>(ptr, {static_cast<size_t>(size)}, owner).cast();
    result.attr("setflags")(false);
    return result;
}

nb::object snapshot(const VecRealView *v) {
    return v ? snapshot(v->data(), v->m()) : nb::none();
}

void copy_vector(const Array &a, Scalar *out, Index size) {
    if (a.ndim() != 1 || a.shape(0) != static_cast<size_t>(size))
        throw nb::value_error("Internal callback vector shape mismatch");
    if (size) std::copy_n(a.data(), size, out);
}

void pack_matrix(const Array &a, MAT *out, Index rows, Index cols) {
    if (a.ndim() != 2 || a.shape(0) != static_cast<size_t>(rows) ||
        a.shape(1) != static_cast<size_t>(cols))
        throw nb::value_error("Internal callback matrix shape mismatch");
    // A C-contiguous rows-by-cols array is a column-major cols-by-rows array.
    // Transposed packing writes only the derivative block, leaving the RHS row.
    if (rows && cols)
        blasfeo_pack_tran_dmat(cols, rows, const_cast<Scalar *>(a.data()), cols, out, 0, 0);
}

class PythonOcp final : public OcpAbstract {
public:
    using Cache = std::unordered_map<std::string, std::vector<Array>>;
    nb::object problem;
    std::unordered_set<std::string> batched;
    Cache cache;

    explicit PythonOcp(nb::object p) : problem(std::move(p)) {
        nb::dict dims = nb::cast<nb::dict>(problem.attr("dimensions"));
        nx_ = nb::cast<std::vector<Index>>(dims["nx"]);
        nu_ = nb::cast<std::vector<Index>>(dims["nu"]);
        ng_ = nb::cast<std::vector<Index>>(dims["ng"]);
        ni_ = nb::cast<std::vector<Index>>(dims["ng_ineq"]);
        tx_ = nb::cast<std::vector<Index>>(dims["nx_tangent"]);
        tu_ = nb::cast<std::vector<Index>>(dims["nu_tangent"]);
        for (auto item : nb::cast<nb::dict>(problem.attr("batched")))
            batched.insert(nb::cast<std::string>(item.first));
    }

    Index get_nx(Index k) const override { return nx_[k]; }
    Index get_nu(Index k) const override { return nu_[k]; }
    Index get_ng(Index k) const override { return ng_[k]; }
    Index get_ng_ineq(Index k) const override { return ni_[k]; }
    Index get_nx_tangent(Index k) const override { return tx_[k]; }
    Index get_nu_tangent(Index k) const override { return tu_[k]; }
    Index get_horizon_length() const override { return static_cast<Index>(nx_.size()); }

    void intermediate(const IpData<OcpType> &data) {
        // Fatrop passes const data, but its diagnostic getters lazily fill caches
        // on the actual mutable iterate, as they do for native iteration output.
        auto &iterate = const_cast<IpIterate<OcpType> &>(data.current_iterate());
        nb::dict stats;
        stats["objective"] = iterate.obj_value();
        stats["inf_pr"] = norm_inf(iterate.constr_viol());
        stats["inf_du"] = std::max(iterate.dual_infeasibility_x_inf(),
                                   iterate.dual_infeasibility_s_inf());
        stats["mu"] = iterate.mu();
        problem.attr("_intermediate")(
            data.iteration_number(), snapshot(&iterate.primal_x()), stats);
    }

    template <typename MakeArgs>
    Array evaluate(const char *name, Index k, MakeArgs make_args) const {
        auto found = cache.find(name);
        if (found != cache.end()) return found->second.at(k);
        return nb::cast<Array>(problem.attr("_evaluate")(name, k, make_args()));
    }

    Index eval_BAbt(const Scalar *xp, const Scalar *u, const Scalar *x, MAT *out, Index k) override {
        auto a = evaluate("eval_BAt", k, [&] { return nb::make_tuple(
            snapshot(xp, nx_[k+1]), snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        pack_matrix(a, out, tu_[k] + tx_[k], tx_[k+1]);
        return 0;
    }

    Index eval_RSQrqt(const Scalar *scale, const Scalar *u, const Scalar *x,
                     const Scalar *ld, const Scalar *le, const Scalar *li, MAT *out, Index k) override {
        auto a = evaluate("eval_RSQt", k, [&] { return nb::make_tuple(
            *scale, snapshot(u, nu_[k]), snapshot(x, nx_[k]),
            snapshot(ld, k+1 < get_horizon_length() ? tx_[k+1] : 0),
            snapshot(le, ng_[k]), snapshot(li, ni_[k])); });
        pack_matrix(a, out, tu_[k] + tx_[k], tu_[k] + tx_[k]);
        return 0;
    }

    Index eval_Ggt(const Scalar *u, const Scalar *x, MAT *out, Index k) override {
        auto a = evaluate("eval_Gt", k, [&] { return nb::make_tuple(
            snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        pack_matrix(a, out, tu_[k] + tx_[k], ng_[k]);
        return 0;
    }

    Index eval_Ggt_ineq(const Scalar *u, const Scalar *x, MAT *out, Index k) override {
        auto a = evaluate("eval_Gt_ineq", k, [&] { return nb::make_tuple(
            snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        pack_matrix(a, out, tu_[k] + tx_[k], ni_[k]);
        return 0;
    }

    Index eval_b(const Scalar *xp, const Scalar *u, const Scalar *x, Scalar *out, Index k) override {
        auto a = evaluate("eval_b", k, [&] { return nb::make_tuple(
            snapshot(xp, nx_[k+1]), snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        copy_vector(a, out, tx_[k+1]);
        return 0;
    }

    Index eval_g(const Scalar *u, const Scalar *x, Scalar *out, Index k) override {
        auto a = evaluate("eval_g", k, [&] { return nb::make_tuple(
            snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        copy_vector(a, out, ng_[k]);
        return 0;
    }

    Index eval_gineq(const Scalar *u, const Scalar *x, Scalar *out, Index k) override {
        auto a = evaluate("eval_gineq", k, [&] { return nb::make_tuple(
            snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        copy_vector(a, out, ni_[k]);
        return 0;
    }

    Index eval_rq(const Scalar *scale, const Scalar *u, const Scalar *x, Scalar *out, Index k) override {
        auto a = evaluate("eval_rq", k, [&] { return nb::make_tuple(
            *scale, snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        copy_vector(a, out, tu_[k] + tx_[k]);
        return 0;
    }

    Index eval_L(const Scalar *scale, const Scalar *u, const Scalar *x, Scalar *out, Index k) override {
        auto a = evaluate("eval_L", k, [&] { return nb::make_tuple(
            *scale, snapshot(u, nu_[k]), snapshot(x, nx_[k])); });
        if (a.ndim() != 0) throw nb::value_error("Internal cost shape mismatch");
        *out = *a.data();
        return 0;
    }

    Index get_bounds(Scalar *lower, Scalar *upper, Index k) const override {
        auto a = evaluate("get_bounds", k, [] { return nb::make_tuple(); });
        if (a.ndim() != 2 || a.shape(0) != 2 || a.shape(1) != static_cast<size_t>(ni_[k]))
            throw nb::value_error("Internal bounds shape mismatch");
        if (ni_[k]) {
            std::copy_n(a.data(), ni_[k], lower);
            std::copy_n(a.data() + ni_[k], ni_[k], upper);
        }
        return 0;
    }

    Index get_initial_xk(Scalar *out, Index k) const override {
        auto a = evaluate("get_initial_xk", k, [] { return nb::make_tuple(); });
        copy_vector(a, out, nx_[k]);
        return 0;
    }

    Index get_initial_uk(Scalar *out, Index k) const override {
        auto a = evaluate("get_initial_uk", k, [] { return nb::make_tuple(); });
        copy_vector(a, out, nu_[k]);
        return 0;
    }

    void apply_retraction_xk(Index k, const Scalar *x, const Scalar *dx, Scalar alpha,
                             Scalar *out) override {
        auto a = evaluate("apply_retraction_xk", k, [&] { return nb::make_tuple(
            snapshot(x, nx_[k]), snapshot(dx, tx_[k]), alpha); });
        copy_vector(a, out, nx_[k]);
    }

    void apply_retraction_uk(Index k, const Scalar *u, const Scalar *du, Scalar alpha,
                             Scalar *out) override {
        auto a = evaluate("apply_retraction_uk", k, [&] { return nb::make_tuple(
            snapshot(u, nu_[k]), snapshot(du, tu_[k]), alpha); });
        copy_vector(a, out, nu_[k]);
    }

private:
    std::vector<Index> nx_, nu_, ng_, ni_, tx_, tu_;
};

// A cache belongs to one horizon operation, including during exception unwinding.
class BatchScope {
public:
    BatchScope(PythonOcp &ocp, std::initializer_list<const char *> families,
               const VecRealView *primal = nullptr, const VecRealView *lam = nullptr,
               Scalar scale = 1., const VecRealView *delta = nullptr, Scalar alpha = 0.)
        : ocp_(ocp), previous_(std::move(ocp.cache)) {
        try {
            ocp.cache.clear();
            nb::list names;
            for (const char *name : families)
                if (ocp.batched.count(name)) names.append(name);
            if (!names.empty()) {
                auto prepared = nb::cast<nb::dict>(ocp.problem.attr("_prepare")(
                    names, snapshot(primal), snapshot(lam), scale, snapshot(delta), alpha));
                for (auto item : prepared)
                    ocp.cache.emplace(nb::cast<std::string>(item.first),
                                      nb::cast<std::vector<Array>>(item.second));
            }
        } catch (...) {
            ocp_.cache = std::move(previous_);
            throw;
        }
    }
    ~BatchScope() { ocp_.cache = std::move(previous_); }
    BatchScope(const BatchScope &) = delete;
    BatchScope &operator=(const BatchScope &) = delete;
private:
    PythonOcp &ocp_;
    PythonOcp::Cache previous_;
};

class PythonNlp final : public NlpOcp {
public:
    explicit PythonNlp(std::shared_ptr<PythonOcp> ocp) : NlpOcp(ocp), ocp_(std::move(ocp)) {}

    void callback(const IpData<OcpType> &data) override {
        ocp_->intermediate(data);
    }

    Index eval_lag_hess(const Info &i, Scalar scale, const VecRealView &x, const VecRealView &s,
                       const VecRealView &lam, Hessian<OcpType> &out) override {
        BatchScope scope(*ocp_, {"eval_RSQt"}, &x, &lam, scale);
        return NlpOcp::eval_lag_hess(i, scale, x, s, lam, out);
    }
    Index eval_constr_jac(const Info &i, const VecRealView &x, const VecRealView &s,
                         Jacobian<OcpType> &out) override {
        BatchScope scope(*ocp_, {"eval_Gt", "eval_Gt_ineq", "eval_BAt"}, &x);
        return NlpOcp::eval_constr_jac(i, x, s, out);
    }
    Index eval_constraint_violation(const Info &i, const VecRealView &x, const VecRealView &s,
                                   VecRealView &out) override {
        BatchScope scope(*ocp_, {"eval_g", "eval_gineq", "eval_b"}, &x);
        return NlpOcp::eval_constraint_violation(i, x, s, out);
    }
    Index eval_objective_gradient(const Info &i, Scalar scale, const VecRealView &x,
                                 const VecRealView &s, VecRealView &gx, VecRealView &gs) override {
        BatchScope scope(*ocp_, {"eval_rq"}, &x, nullptr, scale);
        return NlpOcp::eval_objective_gradient(i, scale, x, s, gx, gs);
    }
    Index eval_objective(const Info &i, Scalar scale, const VecRealView &x,
                        const VecRealView &s, Scalar &out) override {
        BatchScope scope(*ocp_, {"eval_L"}, &x, nullptr, scale);
        return NlpOcp::eval_objective(i, scale, x, s, out);
    }
    Index get_bounds(const Info &i, VecRealView &lo, VecRealView &hi) override {
        BatchScope scope(*ocp_, {"get_bounds"});
        return NlpOcp::get_bounds(i, lo, hi);
    }
    Index get_initial_primal(const Info &i, VecRealView &out) override {
        BatchScope scope(*ocp_, {"get_initial_xk", "get_initial_uk"});
        return NlpOcp::get_initial_primal(i, out);
    }
    void apply_retraction(const Info &i, const VecRealView &x, const VecRealView &dx,
                          Scalar alpha, VecRealView &out) override {
        BatchScope scope(*ocp_, {"apply_retraction_xk", "apply_retraction_uk"}, &x,
                         nullptr, 1., &dx, alpha);
        NlpOcp::apply_retraction(i, x, dx, alpha, out);
    }
private:
    std::shared_ptr<PythonOcp> ocp_;
};

void set_options(OptionRegistry &registry, const nb::dict &options) {
    for (auto item : options) {
        auto name = nb::cast<std::string>(item.first);
        nb::handle value = item.second;
        const auto type = registry.get_option_type(name);
        if (type == OptionTypeId::Unknown)
            throw nb::value_error(("Unknown Fatrop option: " + name).c_str());
        try {
            if (type == OptionTypeId::Scalar &&
                (nb::isinstance<nb::float_>(value) || nb::isinstance<nb::int_>(value)))
                registry.set_option(name, nb::cast<Scalar>(value));
            else if (nb::isinstance<nb::bool_>(value))
                registry.set_option(name, nb::cast<bool>(value));
            else if (nb::isinstance<nb::int_>(value))
                registry.set_option(name, nb::cast<Index>(value));
            else if (nb::isinstance<nb::float_>(value))
                registry.set_option(name, nb::cast<Scalar>(value));
            else if (nb::isinstance<nb::str>(value))
                registry.set_option(name, nb::cast<std::string>(value));
            else
                throw nb::type_error(("Unsupported type for Fatrop option: " + name).c_str());
        } catch (const std::runtime_error &e) {
            throw nb::value_error(e.what());
        }
    }
}

class NativeSolver {
public:
    NativeSolver(nb::object problem, nb::dict options)
        : ocp_(std::make_shared<PythonOcp>(std::move(problem))),
          nlp_(std::make_shared<PythonNlp>(ocp_)), options_(std::move(options)) {}

    int traverse(visitproc visit, void *arg) const {
        if (ocp_) Py_VISIT(ocp_->problem.ptr());
        Py_VISIT(options_.ptr());
        return 0;
    }

    void clear() {
        nlp_.reset();
        ocp_.reset();
        options_.reset();
    }

    nb::dict solve() {
        if (!nlp_) throw std::runtime_error("Native solver has been garbage-collected");
        // Build lazily: inspecting a problem never invokes numerical callbacks.
        // Fresh algorithm state also makes retrying after callback exceptions safe.
        OptionRegistry registry;
        IpAlgBuilder<OcpType> builder(nlp_);
        auto algorithm = builder.with_options_registry(&registry).build();
        // These two upstream options are process globals. Reapply their native
        // defaults so an earlier Solver's options do not change this one's defaults.
        registry.set_option("print_level", Index(5));
        registry.set_option("suppress_banner", false);
        set_options(registry, options_);
        auto start = std::chrono::steady_clock::now();
        auto status = algorithm->optimize();
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        auto data = builder.get_ipdata();
        auto &timing = data->timing_statistics();
        nb::dict stats;
        stats["iterations"] = data->iteration_number();
        stats["solve_time"] = elapsed;
        stats["initialization_time"] = timing.initialization.elapsed();
        stats["eval_objective_time"] = timing.eval_objective.elapsed();
        stats["eval_gradient_time"] = timing.eval_gradient.elapsed();
        stats["eval_constraints_time"] = timing.eval_constraint_violation.elapsed();
        stats["eval_jacobian_time"] = timing.eval_jacobian.elapsed();
        stats["eval_hessian_time"] = timing.eval_hessian.elapsed();
        nb::dict result;
        result["status"] = nb::cast(status);
        result["primal"] = snapshot(&algorithm->solution_primal());
        result["objective"] = data->current_iterate().obj_value();
        result["stats"] = stats;
        return result;
    }
private:
    std::shared_ptr<PythonOcp> ocp_;
    std::shared_ptr<PythonNlp> nlp_;
    nb::dict options_;
};

// Callbacks can capture their Solver. Expose native-held Python references so
// Python can discover and collect cycles crossing the language boundary.
int native_solver_traverse(PyObject *self, visitproc visit, void *arg) {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self)) return 0;
    return nb::inst_ptr<NativeSolver>(self)->traverse(visit, arg);
}

int native_solver_clear(PyObject *self) {
    if (nb::inst_ready(self)) nb::inst_ptr<NativeSolver>(self)->clear();
    return 0;
}

PyType_Slot native_solver_slots[] = {
    {Py_tp_traverse, reinterpret_cast<void *>(native_solver_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(native_solver_clear)},
    {0, nullptr}
};
} // namespace

NB_MODULE(_native, m) {
    m.doc() = "Native Fatrop OcpAbstract adapter and horizon-wide batching";
    m.attr("FATROP_VERSION") = "1.2.0";
    m.attr("FATROP_COMMIT") = "86929bad166f49109133a1bf7a5ae6ecb343b2a3";
    m.attr("BLASFEO_COMMIT") = "5cb0a065895479e7dc54c1771fec48548aec0746";
    nb::enum_<IpSolverReturnFlag>(m, "Status")
        .value("Success", IpSolverReturnFlag::Success)
        .value("MaxIterExceeded", IpSolverReturnFlag::MaxIterExceeded)
        .value("StopAtAcceptablePoint", IpSolverReturnFlag::StopAtAcceptablePoint)
        .value("LocalInfeasibility", IpSolverReturnFlag::LocalInfeasibility)
        .value("LineSearchFailed", IpSolverReturnFlag::LineSearchFailed)
        .value("ErrorInStepComputation", IpSolverReturnFlag::ErrorInStepComputation)
        .value("Unknown", IpSolverReturnFlag::Unknown);
    nb::class_<NativeSolver>(m, "NativeSolver", nb::type_slots(native_solver_slots))
        .def(nb::init<nb::object, nb::dict>())
        .def("solve", &NativeSolver::solve);
}
