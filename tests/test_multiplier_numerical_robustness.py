''' Regression tests for multiplier models whose optimal weights are not
    unique and unbounded along the optimal face (GitHub issue #2 of
    Donut-the-1st/pyDEA). CBC 2.10 returned weights of about 1e10 for DMUs
    D and E of the small test data, which, reported with about 8
    significant digits, no longer satisfied the constraints or reproduced
    the optimal objective value.
'''
import logging

import pulp
import pytest

from pyDEA.core.models.multiplier_model_base import MultiplierModelBase
from pyDEA.core.models.multiplier_model import MultiplierInputOrientedModel
from pyDEA.core.models.multiplier_model import MultiplierOutputOrientedModel
from pyDEA.core.models.multiplier_model_decorators import (
    MultiplierModelVRSDecorator, MultiplierModelWithDisposableCategories)
from pyDEA.core.models.multiplier_solution_check import certified_dual_bound
from pyDEA.core.models.multiplier_solution_check import check_primal
from pyDEA.core.models.super_efficiency_model import SupperEfficiencyModel
import pyDEA.core.models.multiplier_model_base as multiplier_model_base
from pyDEA.core.utils import dea_utils

from tests.test_CRS_env_input_oriented_model import data
import tests.utils_for_tests as utils_for_tests

DMUS = ['A', 'B', 'C', 'D', 'E']
MAX_ABS_WEIGHT = 1e3


def _concrete_model(orientation):
    if orientation == 'output':
        return MultiplierOutputOrientedModel()
    return MultiplierInputOrientedModel()


def _build_model(data, orientation, vrs, free_categories=()):
    tolerance = 1e-12 if free_categories else 0
    model = MultiplierModelBase(data, tolerance, _concrete_model(orientation))
    if free_categories:
        model = MultiplierModelWithDisposableCategories(
            model, set(free_categories))
    if vrs:
        model = MultiplierModelVRSDecorator(model)
    return model


def _solve_for_dmu(model, data, dmu):
    ''' Builds and solves the LP of a given DMU without filling a solution.
    '''
    model._create_lp()
    model._update_lp(data._DMU_user_name_to_code[dmu])
    model.lp_model.solve(dea_utils.get_lp_solver())
    assert model.lp_model.status == pulp.LpStatusOptimal
    return model.lp_model


def _set_values(model, values, intercept=None):
    for category, value in values.items():
        if category in model._input_variables:
            model._input_variables[category].varValue = value
        else:
            model._output_variables[category].varValue = value
    if intercept is not None:
        model._vrs_variable.varValue = intercept


MODELS = [
    # the three cases of issue #2
    ('output', True, ()),
    ('output', True, ('q',)),
    ('output', False, ('x1',)),
    # counterparts
    ('output', False, ()),
    ('input', True, ()),
    ('input', True, ('q',)),
    ('input', False, ('x1',)),
    ('input', False, ()),
]


@pytest.mark.parametrize('orientation, vrs, free_categories', MODELS)
def test_all_dmus_have_valid_bounded_weights(data, orientation, vrs,
                                             free_categories):
    model = _build_model(data, orientation, vrs, free_categories)
    model_solution = model.run()
    utils_for_tests.check_multiplier_solution_consistent(
        model_solution, data, DMUS, orientation, vrs=vrs,
        free_categories=free_categories)
    for dmu in DMUS:
        dmu_code = data._DMU_user_name_to_code[dmu]
        values = (list(model_solution.input_duals[dmu_code].values()) +
                  list(model_solution.output_duals[dmu_code].values()))
        if vrs:
            values.append(model_solution.get_VRS_dual(dmu_code))
        assert max(abs(value) for value in values) <= MAX_ABS_WEIGHT


def test_efficiency_scores_of_issue_cases_are_unchanged(data):
    expected = {
        ('output', True, ()): [0.5, 1, 1, 1, 1],
        ('output', True, ('q',)): [0.5, 1, 1, 1, 1],
        ('output', False, ('x1',)): [0.5, 1, 0.8333333458, 0.7142857143, 1],
    }
    for (orientation, vrs, free_categories), scores in expected.items():
        model = _build_model(data, orientation, vrs, free_categories)
        model_solution = model.run()
        utils_for_tests.check_efficiency_scores(DMUS, scores, model_solution,
                                                data)


def test_check_primal_detects_values_reported_by_cbc(data):
    model = _build_model(data, 'output', True)
    lp_model = _solve_for_dmu(model, data, 'E')

    # VRS, DMU E: values from issue #2; objective recomputes to 0 instead
    # of 1 and the constraint of E is violated by 1
    _set_values(model, {'x1': 0, 'x2': 1e10, 'q': 0.5}, -2e10)
    is_ok, info = check_primal(lp_model, 1.0)
    assert not is_ok
    assert info['gap'] >= 0.5
    assert info['violation'] >= 0.5

    # a valid optimal solution passes
    _set_values(model, {'x1': 0.0625, 'x2': 0.125, 'q': 0.5}, 0.375)
    is_ok, info = check_primal(lp_model, 1.0)
    assert is_ok


def test_check_primal_detects_extreme_but_consistent_values(data):
    model = _build_model(data, 'output', True)
    lp_model = _solve_for_dmu(model, data, 'D')
    # VRS, DMU D: values from issue #2 reproduce the objective and satisfy
    # all constraints, but only by cancellation of terms of about 1e10
    _set_values(model, {'x1': 0.33333333, 'x2': 1e10, 'q': 1}, -2e10)
    is_ok, info = check_primal(lp_model, 1.0)
    assert not is_ok
    assert info['cancellation'] > 1e9

    _set_values(model, {'x1': 0.33333333, 'x2': 0.66666667, 'q': 1},
                -1.3333333)
    is_ok, info = check_primal(lp_model, 1.0)
    assert is_ok


def test_check_primal_detects_inconsistent_objective_crs(data):
    model = _build_model(data, 'output', False, ('x1',))
    lp_model = _solve_for_dmu(model, data, 'E')
    # CRS with weakly disposable x1, DMU E: objective recomputes to 200
    _set_values(model, {'x1': -3333333300, 'x2': 1e10, 'q': 0.5})
    is_ok, info = check_primal(lp_model, 1.0)
    assert not is_ok
    assert info['objective'] == pytest.approx(200)

    _set_values(model, {'x1': 0.1, 'x2': 0.2, 'q': 0.5})
    is_ok, info = check_primal(lp_model, 1.0)
    assert is_ok


def test_certified_dual_bound(data):
    model = _build_model(data, 'output', True)
    lp_model = _solve_for_dmu(model, data, 'E')
    z_star, scale, is_ok = certified_dual_bound(lp_model)
    assert is_ok
    assert z_star == pytest.approx(1, abs=1e-7)
    assert scale >= 1

    # a dual value with the wrong sign must be detected
    dmu_code = data._DMU_user_name_to_code['E']
    name = [name for name, dmu in model._dmu_constraint_names.items()
            if dmu == dmu_code][0]
    constraint = lp_model.constraints[name]
    assert constraint.pi < 0
    constraint.pi = -constraint.pi
    z_star, scale, is_ok = certified_dual_bound(lp_model)
    assert not is_ok


def test_certified_dual_bound_input_oriented(data):
    model = _build_model(data, 'input', True)
    lp_model = _solve_for_dmu(model, data, 'A')
    z_star, scale, is_ok = certified_dual_bound(lp_model)
    assert is_ok
    is_primal_ok, info = check_primal(lp_model, z_star)
    assert is_primal_ok
    assert info['objective'] == pytest.approx(z_star, abs=1e-7)


def test_unresolved_inconsistency_is_reported(data, monkeypatch, caplog):
    ''' If the solver's weights fail validation and no other optimal
        solution can be selected, this must be reported for each DMU, and
        the efficiency score must come from the certified objective value.
    '''
    info = dict(objective=0.0, gap=1.0, violation=1.0, cancellation=1e10)
    monkeypatch.setattr(multiplier_model_base, 'check_primal',
                        lambda *args, **kwargs: (False, info))
    monkeypatch.setattr(multiplier_model_base, 'select_bounded_solution',
                        lambda *args, **kwargs: False)
    model = _build_model(data, 'output', True)
    with caplog.at_level(logging.WARNING):
        model_solution = model.run()
    for dmu in DMUS:
        warnings = model_solution.get_numerical_warnings(
            data._DMU_user_name_to_code[dmu])
        assert len(warnings) == 1
        assert 'no consistent optimal solution' in warnings[0]
        assert any('DMU ' + dmu + ':' in record.getMessage()
                   for record in caplog.records)
    utils_for_tests.check_efficiency_scores(DMUS, [0.5, 1, 1, 1, 1],
                                            model_solution, data)


def test_masked_efficiency_score_is_reported(data, monkeypatch):
    ''' Reproduces the situation before the fix: the inconsistent weights
        reported for DMU E give objective 200, the efficiency score 1/200 is
        set to 1 by is_efficient because of the positive lambda variable of
        E itself. This must not happen silently.
    '''
    original = MultiplierModelBase._ensure_valid_weights
    e_code = data._DMU_user_name_to_code['E']

    def without_validation(self, dmu_code, model_solution):
        if dmu_code != e_code:
            return original(self, dmu_code, model_solution)
        _set_values(self, {'x1': -3333333300, 'x2': 1e10, 'q': 0.5})
        return None

    monkeypatch.setattr(MultiplierModelBase, '_ensure_valid_weights',
                        without_validation)
    model = _build_model(data, 'output', False, ('x1',))
    model_solution = model.run()
    assert model_solution.get_efficiency_score(e_code) == 1
    warnings = model_solution.get_numerical_warnings(e_code)
    assert len(warnings) == 1
    assert 'inconsistent with the dual solution' in warnings[0]
    for dmu in ['A', 'B', 'C', 'D']:
        assert model_solution.get_numerical_warnings(
            data._DMU_user_name_to_code[dmu]) == []


@pytest.mark.parametrize('vrs', [False, True])
def test_super_efficiency_does_not_report_warnings(data, vrs):
    model = SupperEfficiencyModel(_build_model(data, 'output', vrs))
    model_solution = model.run()
    assert model_solution.get_numerical_warnings() == {}


def test_all_solves_use_get_lp_solver(data, monkeypatch):
    solver = dea_utils.get_lp_solver()
    original_solve = pulp.LpProblem.solve
    solved_problems = []

    def checked_solve(self, solver_arg=None, **kwargs):
        assert solver_arg is solver
        solved_problems.append(self.name)
        return original_solve(self, solver_arg, **kwargs)

    monkeypatch.setattr(pulp.LpProblem, 'solve', checked_solve)
    model = _build_model(data, 'output', True)
    model.run()
    assert len(solved_problems) >= len(DMUS)
