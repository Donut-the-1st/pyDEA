from pulp import LpStatusOptimal
import pytest

from pyDEA.core.data_processing.read_data import read_data
from pyDEA.core.data_processing.read_data import construct_input_data_instance
from pyDEA.core.data_processing.read_data import validate_data
from pyDEA.core.data_processing.read_data import convert_to_dictionary
from pyDEA.core.utils.dea_utils import clean_up_pickled_files


EQ_TOLERANCE = 1e-10
WEIGHT_RESTRICTION_TOLERANCE = 1e-7


@pytest.fixture
def DEA_example2_data(request):
    categories, xls_data, dmu_name, sheet_name = read_data(
        'tests/DEA_example2_data.xlsx')
    coefficients, has_same_dmus = convert_to_dictionary(xls_data)
    assert has_same_dmus is False
    assert validate_data(categories, coefficients) is True
    request.addfinalizer(clean_up_pickled_files)
    return construct_input_data_instance(categories, coefficients)


@pytest.fixture
def categorical_from_book(request):
    categories, xls_data, dmu_name, sheet_name = read_data(
        'tests/categorical_test_from_book.xlsx')
    coefficients, has_same_dmus = convert_to_dictionary(xls_data)
    assert has_same_dmus is False
    assert validate_data(categories, coefficients) is True
    data = construct_input_data_instance(categories, coefficients)
    data.add_input_category('Area')
    data.add_input_category('Books')
    data.add_input_category('Staff')
    data.add_output_category('Regist')
    data.add_output_category('Borrow')
    request.addfinalizer(clean_up_pickled_files)
    return data


@pytest.fixture
def DEA_example_data(request):
    categories, xls_data, dmu_name, sheet_name = read_data(
        'tests/DEA_example_data.xlsx')
    coefficients, has_same_dmus = convert_to_dictionary(xls_data)
    assert has_same_dmus is False
    assert validate_data(categories, coefficients) is True
    request.addfinalizer(clean_up_pickled_files)
    return construct_input_data_instance(categories, coefficients)


@pytest.fixture
def categorical_data(request):
    categories, xls_data, dmu_name, sheet_name = read_data(
        'tests/categorical_test.xlsx')
    coefficients, has_same_dmus = convert_to_dictionary(xls_data)
    assert has_same_dmus is False
    assert validate_data(categories, coefficients) is True
    request.addfinalizer(clean_up_pickled_files)
    return construct_input_data_instance(categories, coefficients)


def check_onion_ranks(input_data, dmus, expected_ranks, ranks):
    for count, dmu in enumerate(dmus):
        print('dmu', dmu, 'rank', ranks[input_data._DMU_user_name_to_code[dmu]],
              'expected_ranks', expected_ranks[count])
        assert ranks[input_data._DMU_user_name_to_code[
            dmu]] == expected_ranks[count]


def check_lambda_variables(dmu1, dmu2, value, model_solution, data):
    ''' (str, str, double, Solution, InputData) -> NoneType
    '''
    dmu1_code = data._DMU_user_name_to_code[dmu1]
    dmu2_code = data._DMU_user_name_to_code[dmu2]
    lambda_var = model_solution.get_lambda_variables(
        dmu1_code).get(dmu2_code, 0)
    assert abs(lambda_var - value) <= EQ_TOLERANCE


def check_categories_for_dmu(dmu, categories, expected_values,
                             model_solution, data):
    dmu_code = data._DMU_user_name_to_code[dmu]
    for count, category in enumerate(categories):
        if category in data.input_categories:
            dual_var = model_solution.get_input_dual(dmu_code, category)
        elif category in data.output_categories:
            dual_var = model_solution.get_output_dual(dmu_code, category)
        assert abs(dual_var - expected_values[count]) <= EQ_TOLERANCE


def check_efficiency_scores(dmus, expected_scores, model_solution,
                            data, tolerance=EQ_TOLERANCE):
    for count, dmu in enumerate(dmus):
        dmu_code = data._DMU_user_name_to_code[dmu]
        print('DMU: {0}, score: {1}'.format(
            dmu, model_solution.efficiency_scores[dmu_code]))
        assert abs(model_solution.efficiency_scores[
                   dmu_code] - expected_scores[count]) <= tolerance


def check_optimal_solution_status_and_sizes(model_solution, data):
    for dmu_code in data.DMU_codes:
        assert model_solution.lp_status[dmu_code] == LpStatusOptimal
        assert len(model_solution.input_duals[dmu_code]) == len(
            data.input_categories)
        assert len(model_solution.output_duals[dmu_code]) == len(
            data.output_categories)


def check_VRS_duals(dmus, expected_values, model_solution, data):
    for count, dmu in enumerate(dmus):
        assert model_solution.get_VRS_dual(data._DMU_user_name_to_code[
                                           dmu]) == expected_values[count]


MULTIPLIER_CONSISTENCY_TOLERANCE = 1e-6
MAX_CANCELLATION_RATIO = 1e4


def check_multiplier_solution_consistent(
        model_solution, data, dmus, orientation, vrs=False,
        free_categories=(), tolerance=MULTIPLIER_CONSISTENCY_TOLERANCE,
        max_cancellation=MAX_CANCELLATION_RATIO):
    ''' Checks that the weights (and VRS intercept) reported for each given
        DMU form a valid optimal solution of the basic multiplier model,
        whatever particular optimal solution the solver selected:
        normalisation, objective value consistent with the efficiency
        score, all DMU constraints, signs of weights of categories that are
        not weakly disposable, no large cancellations (which would indicate
        extreme values that lost their precision), and no numerical
        warnings.

        Only for models without weight restrictions and non-discretionary
        categories.

        Args:
            model_solution (Solution): solution to check.
            data (InputData): input data.
            dmus (list of str): DMU user names to check.
            orientation (str): 'input' or 'output'.
            vrs (bool): True if the model has a VRS intercept.
            free_categories (iterable of str): weakly disposable
                categories, whose weights are free.
            tolerance (double): absolute tolerance.
            max_cancellation (double): maximum ratio between the sum of
                absolute terms of a constraint and its value.
    '''
    assert orientation in ('input', 'output')

    def weighted_sum(dmu_code, categories, weights):
        terms = [data.coefficients[dmu_code, category] * weights[category]
                 for category in categories]
        return sum(terms), sum(abs(term) for term in terms)

    def check_cancellation(value, abs_terms, what):
        assert abs_terms <= max_cancellation * max(1.0, abs(value)), (
            '{0}: sum of absolute terms {1} is too large for value {2}'
            .format(what, abs_terms, value))

    for dmu in dmus:
        dmu_code = data._DMU_user_name_to_code[dmu]
        assert model_solution.lp_status[dmu_code] == LpStatusOptimal
        input_weights = {category: model_solution.get_input_dual(
            dmu_code, category) for category in data.input_categories}
        output_weights = {category: model_solution.get_output_dual(
            dmu_code, category) for category in data.output_categories}
        intercept = model_solution.get_VRS_dual(dmu_code) if vrs else 0
        weights = dict(input_weights)
        weights.update(output_weights)
        for category, weight in weights.items():
            if category not in free_categories:
                assert weight >= -tolerance, (
                    'DMU {0}: weight of {1} is negative: {2}'.format(
                        dmu, category, weight))

        virtual_input, abs_input = weighted_sum(
            dmu_code, data.input_categories, input_weights)
        virtual_output, abs_output = weighted_sum(
            dmu_code, data.output_categories, output_weights)
        efficiency = model_solution.get_efficiency_score(dmu_code)
        if orientation == 'output':
            normalisation, abs_normalisation = virtual_output, abs_output
            objective = virtual_input + intercept
            abs_objective = abs_input + abs(intercept)
            expected_objective = 1.0 / efficiency
        else:
            normalisation, abs_normalisation = virtual_input, abs_input
            objective = virtual_output + intercept
            abs_objective = abs_output + abs(intercept)
            expected_objective = efficiency
        assert abs(normalisation - 1) <= tolerance, (
            'DMU {0}: normalisation is {1}'.format(dmu, normalisation))
        check_cancellation(normalisation, abs_normalisation,
                           'DMU {0} normalisation'.format(dmu))
        assert abs(objective - expected_objective) <= tolerance * max(
            1.0, abs(expected_objective)), (
            'DMU {0}: objective recomputed from weights is {1}, efficiency '
            'score corresponds to {2}'.format(dmu, objective,
                                              expected_objective))
        check_cancellation(objective, abs_objective,
                           'DMU {0} objective'.format(dmu))

        for other_code in data.DMU_codes:
            other_input, other_abs_input = weighted_sum(
                other_code, data.input_categories, input_weights)
            other_output, other_abs_output = weighted_sum(
                other_code, data.output_categories, output_weights)
            if orientation == 'output':
                row = other_output - other_input - intercept
            else:
                row = other_output - other_input + intercept
            other = data.get_dmu_user_name(other_code)
            assert row <= tolerance, (
                'DMU {0}: constraint of DMU {1} is violated by {2}'.format(
                    dmu, other, row))
            check_cancellation(row, other_abs_input + other_abs_output +
                               abs(intercept), 'DMU {0}, constraint of DMU {1}'
                               .format(dmu, other))

        assert model_solution.get_numerical_warnings(dmu_code) == []


def _check_lower_bound(dual_value, lower_bound):
    assert dual_value + WEIGHT_RESTRICTION_TOLERANCE >= lower_bound


def _check_upper_bound(dual_value, upper_bound):
    assert dual_value - WEIGHT_RESTRICTION_TOLERANCE <= upper_bound


def _helper_function_for_checking_limits(category, model_solution,
                                         categories, bound, get_dual_func,
                                         bool_func,
                                         process_bound):
    if category in categories:
        for dmu_code in model_solution._input_data.DMU_codes:
            if model_solution.lp_status[dmu_code] == LpStatusOptimal:
                dual_value = get_dual_func(dmu_code, category)
                coeff = model_solution._input_data.coefficients[
                    dmu_code, category]
                bool_func(dual_value, process_bound(bound, coeff))


def _check_if_category_is_within_limits(model_solution, bounds, process_bound):
    for category, (lower_bound, upper_bound) in bounds.items():
        if lower_bound:
            _helper_function_for_checking_limits(
                category, model_solution,
                model_solution._input_data.input_categories,
                lower_bound, model_solution.get_input_dual,
                _check_lower_bound, process_bound)

            _helper_function_for_checking_limits(
                category, model_solution,
                model_solution._input_data.output_categories,
                lower_bound, model_solution.get_output_dual,
                _check_lower_bound, process_bound)

        if upper_bound:
            _helper_function_for_checking_limits(
                category, model_solution,
                model_solution._input_data.input_categories,
                upper_bound, model_solution.get_input_dual,
                _check_upper_bound, process_bound)

            _helper_function_for_checking_limits(
                category, model_solution,
                model_solution._input_data.output_categories,
                upper_bound, model_solution.get_output_dual,
                _check_upper_bound, process_bound)


def _process_abs_bounds(bound, coeff):
    return bound


def check_if_category_is_within_abs_limits(model_solution, bounds):
    _check_if_category_is_within_limits(
        model_solution, bounds, _process_abs_bounds)


def _process_virtual_bounds(bound, coeff):
    return bound / coeff


def check_if_category_is_within_virtual_limits(model_solution, bounds):
    _check_if_category_is_within_limits(
        model_solution, bounds, _process_virtual_bounds)


def _get_proper_dual(category, model_solution, dmu_code):
    if category in model_solution._input_data.input_categories:
        return model_solution.get_input_dual(dmu_code, category)
    elif category in model_solution._input_data.output_categories:
        return model_solution.get_output_dual(dmu_code, category)
    return None


def check_if_category_is_within_price_ratio_constraints(model_solution, bounds):
    for (category_in_nom, category_in_denom), (lower_bound, upper_bound) in bounds.items():
        if lower_bound:
            for dmu_code in model_solution._input_data.DMU_codes:
                if model_solution.lp_status[dmu_code] == LpStatusOptimal:
                    dual_value = _get_proper_dual(
                        category_in_nom, model_solution, dmu_code)
                    dual_value_denom = _get_proper_dual(
                        category_in_denom, model_solution, dmu_code)
                    assert dual_value + WEIGHT_RESTRICTION_TOLERANCE >= lower_bound * dual_value_denom

        if upper_bound:
            for dmu_code in model_solution._input_data.DMU_codes:
                if model_solution.lp_status[dmu_code] == LpStatusOptimal:
                    dual_value = _get_proper_dual(
                        category_in_nom, model_solution, dmu_code)
                    dual_value_denom = _get_proper_dual(
                        category_in_denom, model_solution, dmu_code)
                    assert dual_value - WEIGHT_RESTRICTION_TOLERANCE <= upper_bound * dual_value_denom
