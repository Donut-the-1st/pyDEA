''' Tests for the second phase of the two-phase (maximise slacks) model.

    Phase 2 fixes theta (input orientation) or phi (output orientation) at its
    phase-1 value and maximises the sum of slacks:

        max  sum(s-) + sum(s+)
        s.t. theta*x_o - sum(lambda*x) - s- = 0   (x_o without theta if output)
             sum(lambda*y) - y_o - s+ = 0         (phi*y_o if output)
             sum(lambda) = 1                      (VRS only)

    Weakly disposable categories keep an equality constraint without slack.
    Results are compared with hand-derived values and with an independent
    reference implementation written directly with PuLP.
'''
import pulp
import pytest

import pyDEA.core.utils.model_factory as factory
from pyDEA.core.data_processing.parameters import Parameters
from pyDEA.core.data_processing.read_data import read_data
from pyDEA.core.data_processing.read_data import convert_to_dictionary
from pyDEA.core.data_processing.read_data import construct_input_data_instance
from pyDEA.core.utils.dea_utils import clean_up_pickled_files

TOL = 1e-6
# Slacks are sums of lambda * data. CBC reports solutions with about 8
# significant digits and phase 2 fixes theta/phi within a relative band of
# 1e-7, so slacks are compared with a tolerance relative to the data of the
# DMU (see _slack_tol).
SLACK_REL_TOL = 1e-6


def _slack_tol(model_input, dmu_code):
    scale = sum(abs(model_input.coefficients[dmu_code, category])
                for category in (list(model_input.input_categories) +
                                 list(model_input.output_categories)))
    return SLACK_REL_TOL * max(1.0, scale)

# Small dataset with known two-phase results (inputs I1, I2; outputs O1, O2).
# C is weakly efficient (dominated by B in I1), D lacks output O2 compared to B.
SMALL_CATEGORIES = ['I1', 'I2', 'O1', 'O2']
SMALL_DATA = {'A': [1, 3, 1, 1],
              'B': [2, 1, 1, 1],
              'C': [3, 1, 1, 1],
              'D': [2, 1, 1, 0.5],
              'E': [4, 4, 1, 1]}

# Hand-derived (efficiency, non-zero slacks) for DMUs whose phase-2 optimum
# is unique.
EXPECTED = {
    ('input', 'CRS'): {'A': (1, {}), 'B': (1, {}), 'C': (1, {'I1': 1}),
                       'D': (1, {'O2': 0.5}), 'E': (5 / 12, {})},
    ('input', 'VRS'): {'A': (1, {}), 'B': (1, {}), 'C': (1, {'I1': 1}),
                       'D': (1, {'O2': 0.5}), 'E': (5 / 12, {})},
    ('output', 'CRS'): {'A': (1, {}), 'B': (1, {}), 'C': (1, {'I1': 1}),
                        'D': (1, {'O2': 0.5}), 'E': (5 / 12, {})},
    ('output', 'VRS'): {'A': (1, {}), 'B': (1, {}), 'C': (1, {'I1': 1}),
                        'D': (1, {'O2': 0.5}),
                        'E': (1, {'I1': 2, 'I2': 3})},
}


@pytest.fixture(autouse=True)
def remove_pickled_files(request):
    request.addfinalizer(clean_up_pickled_files)


def _make_params(inputs, outputs, orientation, rts, max_slacks='yes',
                 weakly_disposable='', super_efficiency=''):
    params = Parameters()
    params.update_parameter('INPUT_CATEGORIES', '; '.join(inputs))
    params.update_parameter('OUTPUT_CATEGORIES', '; '.join(outputs))
    params.update_parameter('DEA_FORM', 'env')
    params.update_parameter('RETURN_TO_SCALE', rts)
    params.update_parameter('ORIENTATION', orientation)
    params.update_parameter('NON_DISCRETIONARY_CATEGORIES', '')
    params.update_parameter('WEAKLY_DISPOSAL_CATEGORIES', weakly_disposable)
    params.update_parameter('USE_SUPER_EFFICIENCY', super_efficiency)
    params.update_parameter('ABS_WEIGHT_RESTRICTIONS', '')
    params.update_parameter('VIRTUAL_WEIGHT_RESTRICTIONS', '')
    params.update_parameter('PRICE_RATIO_RESTRICTIONS', '')
    params.update_parameter('MAXIMIZE_SLACKS', max_slacks)
    params.update_parameter('MULTIPLIER_MODEL_TOLERANCE', '0')
    params.update_parameter('CATEGORICAL_CATEGORY', '')
    params.update_parameter('PEEL_THE_ONION', '')
    return params


def _run(categories, data, inputs, outputs, orientation, rts, **kwargs):
    ''' Builds the model the same way the application does and runs it.

        Returns:
            tuple: (input data, phase-1 solution, phase-2 solution or None,
                    dict that maps DMU names to internal DMU codes)
    '''
    model_input = construct_input_data_instance(categories, data)
    params = _make_params(inputs, outputs, orientation, rts, **kwargs)
    factory.add_input_and_output_categories(params, model_input)
    model = factory.create_model(params, model_input)
    first = model.run()
    second = getattr(model, 'second_solution', None)
    codes = {name: code for code, name in
             model_input.DMU_code_to_user_name.items()}
    return model_input, first, second, codes


def _slacks(model_input, solution, dmu_code, orientation):
    ''' Slacks implied by the stored lambda variables and efficiency score,
        computed the same way as the output sheets do.
    '''
    lambdas = solution.get_lambda_variables(dmu_code)
    efficiency = solution.get_efficiency_score(dmu_code)
    coeff = model_input.coefficients
    slacks = {}
    for category in model_input.input_categories:
        scale = efficiency if orientation == 'input' else 1
        slacks[category] = (scale * coeff[dmu_code, category] -
                            sum(value * coeff[dmu, category]
                                for dmu, value in lambdas.items()))
    for category in model_input.output_categories:
        scale = 1 / efficiency if orientation == 'output' else 1
        slacks[category] = (sum(value * coeff[dmu, category]
                                for dmu, value in lambdas.items()) -
                            scale * coeff[dmu_code, category])
    return slacks


def _reference(categories, data, inputs, outputs, orientation, rts,
               weakly_disposable=(), super_efficiency=False):
    ''' Independent textbook two-phase model.

        Returns:
            dict: DMU name -> (efficiency, total slack, dict of slacks), or
                None for DMUs whose phase 1 is not optimal.
    '''
    def value(dmu, category):
        return data[dmu][categories.index(category)]

    result = {}
    for dmu_o in data:
        peers = [dmu for dmu in data if not (super_efficiency and
                                              dmu == dmu_o)]
        lambdas = {dmu: pulp.LpVariable('lambda_{0}'.format(dmu), 0)
                   for dmu in peers}
        theta = pulp.LpVariable('theta', 0)

        def add_constraints(problem, scale, slack_vars=None):
            for category in inputs:
                lhs = ((scale if orientation == 'input' else 1) *
                       value(dmu_o, category) -
                       pulp.lpSum(lambdas[dmu] * value(dmu, category)
                                  for dmu in peers))
                if category in weakly_disposable:
                    problem += lhs == 0
                elif slack_vars is not None:
                    problem += lhs - slack_vars[category] == 0
                else:
                    problem += lhs >= 0
            for category in outputs:
                lhs = (pulp.lpSum(lambdas[dmu] * value(dmu, category)
                                  for dmu in peers) -
                       (scale if orientation == 'output' else 1) *
                       value(dmu_o, category))
                if category in weakly_disposable:
                    problem += lhs == 0
                elif slack_vars is not None:
                    problem += lhs - slack_vars[category] == 0
                else:
                    problem += lhs >= 0
            if rts == 'VRS':
                problem += pulp.lpSum(lambdas.values()) == 1

        sense = pulp.LpMinimize if orientation == 'input' else pulp.LpMaximize
        phase1 = pulp.LpProblem('phase1', sense)
        phase1 += theta
        add_constraints(phase1, theta)
        phase1.solve(pulp.PULP_CBC_CMD(msg=False))
        if phase1.status != pulp.LpStatusOptimal:
            result[dmu_o] = None
            continue
        theta_star = theta.varValue

        slack_vars = {category: pulp.LpVariable('s_{0}'.format(category), 0)
                      for category in inputs + outputs
                      if category not in weakly_disposable}
        phase2 = pulp.LpProblem('phase2', pulp.LpMaximize)
        phase2 += pulp.lpSum(slack_vars.values())
        add_constraints(phase2, theta_star, slack_vars)
        phase2.solve(pulp.PULP_CBC_CMD(msg=False))
        assert phase2.status == pulp.LpStatusOptimal
        efficiency = theta_star if orientation == 'input' else 1 / theta_star
        slacks = {category: var.varValue for category, var in
                  slack_vars.items()}
        result[dmu_o] = (efficiency, pulp.value(phase2.objective), slacks)
    return result


@pytest.mark.parametrize('orientation', ['input', 'output'])
@pytest.mark.parametrize('rts', ['CRS', 'VRS'])
def test_two_phase_small_hand_checked(orientation, rts):
    model_input, first, second, codes = _run(
        SMALL_CATEGORIES, SMALL_DATA, ['I1', 'I2'], ['O1', 'O2'],
        orientation, rts)
    assert second is not None
    for name, (efficiency, expected_slacks) in EXPECTED[
            (orientation, rts)].items():
        code = codes[name]
        assert second.lp_status[code] == pulp.LpStatusOptimal, name
        assert abs(first.get_efficiency_score(code) - efficiency) < TOL, name
        assert abs(second.get_efficiency_score(code) -
                   first.get_efficiency_score(code)) < TOL, name
        slacks = _slacks(model_input, second, code, orientation)
        tol = _slack_tol(model_input, code)
        for category, slack in slacks.items():
            assert abs(slack - expected_slacks.get(category, 0)) < tol, (
                name, category, slacks)


@pytest.mark.parametrize('orientation', ['input', 'output'])
@pytest.mark.parametrize('rts', ['CRS', 'VRS'])
def test_two_phase_small_matches_reference(orientation, rts):
    inputs, outputs = ['I1', 'I2'], ['O1', 'O2']
    model_input, first, second, codes = _run(
        SMALL_CATEGORIES, SMALL_DATA, inputs, outputs, orientation, rts)
    reference = _reference(SMALL_CATEGORIES, SMALL_DATA, inputs, outputs,
                           orientation, rts)
    _assert_matches_reference(model_input, first, second, codes, reference,
                              orientation)


def _load(file_name):
    categories, xls_data, _, _ = read_data(file_name)
    coefficients, has_same_dmus = convert_to_dictionary(xls_data)
    assert has_same_dmus is False
    return categories, coefficients


def _assert_matches_reference(model_input, first, second, codes, reference,
                              orientation):
    for name, ref in reference.items():
        code = codes[name]
        if ref is None:
            continue
        ref_efficiency, ref_total, ref_slacks = ref
        assert first.lp_status[code] == pulp.LpStatusOptimal, name
        assert second.lp_status[code] == pulp.LpStatusOptimal, name
        assert abs(first.get_efficiency_score(code) - ref_efficiency) < TOL, (
            name)
        slacks = _slacks(model_input, second, code, orientation)
        tol = _slack_tol(model_input, code)
        assert all(slack > -tol for slack in slacks.values()), (name, slacks)
        # total slack is unique even when lambdas have alternative optima
        assert abs(sum(slacks.values()) - ref_total) < tol, (
            name, slacks, ref_slacks)


@pytest.mark.parametrize('orientation', ['input', 'output'])
@pytest.mark.parametrize('rts', ['CRS', 'VRS'])
def test_two_phase_example_data_matches_reference(orientation, rts):
    categories, data = _load('tests/DEA_example_data.xlsx')
    inputs, outputs = ['I1', 'I2'], ['O1', 'O2']
    model_input, first, second, codes = _run(
        categories, data, inputs, outputs, orientation, rts)
    reference = _reference(categories, data, inputs, outputs,
                           orientation, rts)
    _assert_matches_reference(model_input, first, second, codes, reference,
                              orientation)


def test_two_phase_weakly_disposable_matches_reference():
    categories, data = _load('tests/DEA_example2_data.csv')
    inputs, outputs = ['I1', 'I2', 'I3'], ['O1', 'O2']
    weak = ('I2', 'O1')
    model_input, first, second, codes = _run(
        categories, data, inputs, outputs, 'input', 'VRS',
        weakly_disposable='; '.join(weak))
    reference = _reference(categories, data, inputs, outputs, 'input', 'VRS',
                           weakly_disposable=weak)
    _assert_matches_reference(model_input, first, second, codes, reference,
                              'input')
    for name in reference:
        code = codes[name]
        slacks = _slacks(model_input, second, code, 'input')
        tol = _slack_tol(model_input, code)
        for category in weak:
            assert abs(slacks[category]) < tol, (name, category, slacks)


@pytest.mark.parametrize('rts', ['CRS', 'VRS'])
def test_two_phase_does_not_change_phase_one(rts):
    categories, data = _load('tests/DEA_example_data.xlsx')
    inputs, outputs = ['I1', 'I2'], ['O1', 'O2']
    _, with_slacks, _, codes = _run(categories, data, inputs, outputs,
                                    'input', rts)
    _, without_slacks, second, codes_plain = _run(
        categories, data, inputs, outputs, 'input', rts, max_slacks='')
    assert second is None
    for name in data:
        assert abs(with_slacks.get_efficiency_score(codes[name]) -
                   without_slacks.get_efficiency_score(codes_plain[name])
                   ) < TOL, name


def test_two_phase_with_super_efficiency_matches_reference():
    categories, data = _load('tests/DEA_example_data.xlsx')
    inputs, outputs = ['I1', 'I2'], ['O1', 'O2']
    model_input, first, second, codes = _run(
        categories, data, inputs, outputs, 'input', 'VRS',
        super_efficiency='yes')
    reference = _reference(categories, data, inputs, outputs, 'input', 'VRS',
                           super_efficiency=True)
    for name, ref in reference.items():
        code = codes[name]
        if ref is None or first.lp_status[code] != pulp.LpStatusOptimal:
            continue
        assert second.lp_status[code] == pulp.LpStatusOptimal, name
        assert abs(first.get_efficiency_score(code) - ref[0]) < TOL, name
        slacks = _slacks(model_input, second, code, 'input')
        tol = _slack_tol(model_input, code)
        assert all(slack > -tol for slack in slacks.values()), (name, slacks)
        assert abs(sum(slacks.values()) - ref[1]) < tol, (name, slacks, ref)
