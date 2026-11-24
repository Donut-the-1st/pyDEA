''' This module contains functions that check the solution of a multiplier
    model LP returned by the solver and, if necessary, select another,
    numerically well-behaved optimal solution.

    Why this is needed: the optimal face of a multiplier model can be
    unbounded along directions that involve free variables (the VRS
    intercept, weights of weakly disposable categories) or weights of
    categories that are zero for the DMU under consideration. CBC's dual
    simplex may then leave such variables at its internal "fake" bound
    (about 1e10). CBC reports values with only about 8 significant digits,
    so at this magnitude the reported weights no longer satisfy the
    constraints or reproduce the optimal objective value.

    The functions below detect such solutions (check_primal), compute a
    certified optimal objective value from the dual solution
    (certified_dual_bound) and select, among all optimal solutions, one with
    the smallest scaled sum of absolute weights (select_bounded_solution).
    Every point of the optimal face is a valid set of DEA multipliers, so
    this selection does not change efficiency scores; the choice of the
    particular optimal point is arbitrary.
'''
import pulp

from pyDEA.core.utils.dea_utils import get_lp_solver

# relative tolerance for objective gap and constraint violations
FEASIBILITY_TOLERANCE = 1e-5
# maximum allowed ratio between the sum of absolute terms of a row and its
# value; CBC reports about 8 significant digits, so the absolute error of a
# recomputed row is roughly this ratio times 1e-8
MAX_CANCELLATION_RATIO = 1e6
# relative tolerance used when checking signs of dual values
DUAL_SIGN_TOLERANCE = 1e-6
# relative tolerances of the objective bound in the second stage
SECOND_STAGE_TOLERANCES = (1e-9, 1e-7, 1e-5)


def list_constraints(lp_model):
    ''' Returns a list of all constraints of a given LP. Works with PuLP
        versions where LpProblem.constraints is a dictionary and with
        versions where it must be called to obtain a list.

        Args:
            lp_model (pulp.LpProblem): linear program.

        Returns:
            list of pulp.LpConstraint: constraints.
    '''
    constraints = lp_model.constraints
    if callable(constraints):
        return list(constraints())
    return list(constraints.values())


def _sense_sign(lp_model):
    ''' Returns 1 for minimisation problems and -1 for maximisation problems.
    '''
    return 1 if lp_model.sense == pulp.LpMinimize else -1


def certified_dual_bound(lp_model):
    ''' Computes the optimal objective value of a solved LP from its dual
        solution and checks that the dual solution is feasible, which makes
        the computed value a bound on the optimal objective value (weak
        duality). Reduced costs are recomputed from the dual values instead
        of being taken from the solver.

        Args:
            lp_model (pulp.LpProblem): solved linear program.

        Returns:
            tuple of (double, double, bool): dual objective value, the sum of
                absolute values of its terms (at least 1, used for scaling
                tolerances), and True if the dual solution passed all checks.
    '''
    sign = _sense_sign(lp_model)
    constraints = list_constraints(lp_model)
    max_pi = 1.0
    for constraint in constraints:
        if constraint.pi is not None:
            max_pi = max(max_pi, abs(constraint.pi))
    sign_tolerance = DUAL_SIGN_TOLERANCE * max_pi

    is_ok = True
    terms = []
    column_sums = dict()
    column_abs_sums = dict()
    for constraint in constraints:
        pi = constraint.pi or 0
        if (constraint.sense == pulp.LpConstraintLE and
                sign * pi > sign_tolerance):
            is_ok = False
        if (constraint.sense == pulp.LpConstraintGE and
                sign * pi < -sign_tolerance):
            is_ok = False
        terms.append(pi * (-constraint.constant))
        for var, coeff in constraint.expr.items():
            column_sums[var] = column_sums.get(var, 0) + pi * coeff
            column_abs_sums[var] = (column_abs_sums.get(var, 0) +
                                    abs(pi * coeff))

    variables = set(column_sums.keys()) | set(lp_model.objective.keys())
    for var in variables:
        cost = lp_model.objective.get(var, 0)
        reduced_cost = cost - column_sums.get(var, 0)
        # dual values are reported with limited precision, so the error of
        # a recomputed reduced cost grows with the size of its terms
        cost_tolerance = DUAL_SIGN_TOLERANCE * max(
            1.0, abs(cost) + column_abs_sums.get(var, 0))
        signed_cost = sign * reduced_cost
        if var.lowBound is not None and var.upBound is not None:
            bound = var.lowBound if signed_cost >= 0 else var.upBound
        elif var.lowBound is not None:
            if signed_cost < -cost_tolerance:
                is_ok = False
            bound = var.lowBound
        elif var.upBound is not None:
            if signed_cost > cost_tolerance:
                is_ok = False
            bound = var.upBound
        else:
            if abs(reduced_cost) > cost_tolerance:
                is_ok = False
            bound = 0
        terms.append(reduced_cost * bound)

    z_star = sum(terms) + lp_model.objective.constant
    scale = max(1.0, sum(abs(term) for term in terms))
    return z_star, scale, is_ok


def check_primal(lp_model, z_star=None):
    ''' Checks the variable values of a solved LP: feasibility of all
        constraints and bounds, agreement of the objective value with a given
        reference value, and absence of large cancellations, which indicate
        that the values reported by the solver have lost their precision.

        Args:
            lp_model (pulp.LpProblem): solved linear program.
            z_star (double, optional): reference optimal objective value.
                If None, the objective value is not compared.

        Returns:
            tuple of (bool, dict): True if all checks passed, and a dictionary
                with the objective value recomputed from the variable values
                ('objective'), its relative gap to z_star ('gap'), the worst
                relative violation ('violation') and the worst cancellation
                ratio ('cancellation').
    '''
    info = dict(objective=None, gap=0.0, violation=0.0, cancellation=0.0)
    variables = set(lp_model.objective.keys())
    for constraint in list_constraints(lp_model):
        variables.update(constraint.expr.keys())
    if any(var.varValue is None for var in variables):
        return False, info

    worst_violation = 0.0
    worst_cancellation = 0.0
    for constraint in list_constraints(lp_model):
        activity = sum(coeff * var.varValue
                       for var, coeff in constraint.expr.items())
        abs_terms = sum(abs(coeff * var.varValue)
                        for var, coeff in constraint.expr.items())
        rhs = -constraint.constant
        residual = activity - rhs
        if constraint.sense == pulp.LpConstraintLE:
            violation = max(0.0, residual)
        elif constraint.sense == pulp.LpConstraintGE:
            violation = max(0.0, -residual)
        else:
            violation = abs(residual)
        worst_violation = max(worst_violation,
                              violation / max(1.0, abs(rhs)))
        worst_cancellation = max(worst_cancellation, abs_terms / max(
            1.0, abs(activity), abs(rhs)))

    for var in variables:
        if var.lowBound is not None and var.varValue < var.lowBound:
            worst_violation = max(worst_violation, (var.lowBound -
                                  var.varValue) / max(1.0, abs(var.lowBound)))
        if var.upBound is not None and var.varValue > var.upBound:
            worst_violation = max(worst_violation, (var.varValue -
                                  var.upBound) / max(1.0, abs(var.upBound)))

    objective = (sum(coeff * var.varValue
                     for var, coeff in lp_model.objective.items()) +
                 lp_model.objective.constant)
    abs_objective_terms = sum(abs(coeff * var.varValue)
                              for var, coeff in lp_model.objective.items())
    worst_cancellation = max(worst_cancellation, abs_objective_terms /
                             max(1.0, abs(objective)))

    info['objective'] = objective
    info['violation'] = worst_violation
    info['cancellation'] = worst_cancellation
    is_ok = (worst_violation <= FEASIBILITY_TOLERANCE and
             worst_cancellation <= MAX_CANCELLATION_RATIO)
    if z_star is not None:
        info['gap'] = abs(objective - z_star) / max(1.0, abs(z_star))
        is_ok = is_ok and info['gap'] <= FEASIBILITY_TOLERANCE
    return is_ok, info


def _new_variable(lp_model, name):
    ''' Creates a non-negative continuous variable, using
        LpProblem.add_variable if this version of PuLP provides it.
    '''
    add_variable = getattr(lp_model, 'add_variable', None)
    if add_variable is not None:
        return add_variable(name, lowBound=0)
    return pulp.LpVariable(name, 0)


def select_bounded_solution(lp_model, z_star, scale, variable_scales):
    ''' Among all optimal solutions of a solved LP, selects one with the
        smallest sum of scaled absolute variable values. A second LP is
        built from copies of all constraints of the given LP and a
        constraint that fixes the objective at its optimal value z_star
        (within a small tolerance, which is increased if the second LP
        cannot be solved). The selected values are written into the
        variables of the given LP. If no tolerance yields a valid solution,
        the original variable values are restored.

        Args:
            lp_model (pulp.LpProblem): solved linear program.
            z_star (double): optimal objective value of lp_model.
            scale (double): scale of z_star used for the tolerance, see
                certified_dual_bound.
            variable_scales (dict of pulp.LpVariable to double): positive
                penalty of each variable; variables not in the dictionary
                get penalty 1.

        Returns:
            bool: True if a valid solution was selected.
    '''
    constraints = list_constraints(lp_model)
    variables = set(lp_model.objective.keys())
    for constraint in constraints:
        variables.update(constraint.expr.keys())
    variables = sorted(variables, key=lambda var: var.name)
    original_values = {var: var.varValue for var in variables}

    for relative_tolerance in SECOND_STAGE_TOLERANCES:
        tolerance = relative_tolerance * scale
        second_stage = pulp.LpProblem('Multiplier_model_weight_selection',
                                      pulp.LpMinimize)
        for count, constraint in enumerate(constraints):
            second_stage += (constraint.copy(),
                             'stage_two_{0}'.format(count))
        objective = lp_model.objective.copy()
        if lp_model.sense == pulp.LpMinimize:
            second_stage += (objective <= z_star + tolerance,
                             'stage_two_objective')
        else:
            second_stage += (objective >= z_star - tolerance,
                             'stage_two_objective')
        penalty_terms = []
        for var in variables:
            penalty = variable_scales.get(var, 1.0)
            if penalty <= 0:
                penalty = 1.0
            if var.lowBound is not None and var.lowBound >= 0:
                penalty_terms.append(penalty * var)
            else:
                abs_var = _new_variable(second_stage,
                                        'abs_value_{0}'.format(var.name))
                second_stage += (abs_var - var >= 0,
                                 'abs_pos_{0}'.format(var.name))
                second_stage += (abs_var + var >= 0,
                                 'abs_neg_{0}'.format(var.name))
                penalty_terms.append(penalty * abs_var)
        second_stage += pulp.lpSum(penalty_terms)
        second_stage.solve(get_lp_solver())
        if second_stage.status == pulp.LpStatusOptimal:
            is_ok, info = check_primal(lp_model, z_star)
            if is_ok:
                return True

    for var, value in original_values.items():
        var.varValue = value
    return False
