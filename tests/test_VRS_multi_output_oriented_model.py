import pytest

from pyDEA.core.models.multiplier_model_base import MultiplierModelBase
from pyDEA.core.models.multiplier_model import MultiplierOutputOrientedModel
from pyDEA.core.models.multiplier_model_decorators import MultiplierModelVRSDecorator
from pyDEA.core.models.multiplier_model_decorators import MultiplierModelWithDisposableCategories
from pyDEA.core.data_processing.read_data import read_data
from pyDEA.core.data_processing.read_data import construct_input_data_instance
from pyDEA.core.data_processing.read_data import validate_data

from tests.test_CRS_env_input_oriented_model import data
import tests.utils_for_tests as utils_for_tests


@pytest.fixture
def model(request, data):
    model = MultiplierModelVRSDecorator(
        MultiplierModelBase(data, 0,
                            MultiplierOutputOrientedModel()))
    return model


def test_VRS_multi_output_oriented_small(model, data):
    model_solution = model.run()
    utils_for_tests.check_optimal_solution_status_and_sizes(
        model_solution, data)

    dmus = ['A', 'B', 'C', 'D', 'E']
    utils_for_tests.check_efficiency_scores(
        dmus, [0.5, 1, 1, 1, 1], model_solution, data)

    utils_for_tests.check_lambda_variables('A', 'B', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('B', 'B', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('C', 'C', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('D', 'D', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('E', 'E', 1, model_solution, data)

    # The optimal weights of D and E are not unique and CBC can return
    # extreme values for them, so any valid optimal solution is accepted
    # instead of one particular weight vector (and VRS intercept).
    utils_for_tests.check_multiplier_solution_consistent(
        model_solution, data, dmus, 'output', vrs=True)


def test_VRS_multi_output_oriented_with_weakly_disposable_vars_small(data):
    model = MultiplierModelVRSDecorator(MultiplierModelWithDisposableCategories(
        MultiplierModelBase(data, 1e-12,
                            MultiplierOutputOrientedModel()), set(['q'])))
    model_solution = model.run()
    utils_for_tests.check_optimal_solution_status_and_sizes(
        model_solution, data)
    dmus = ['A', 'B', 'C', 'D', 'E']
    utils_for_tests.check_efficiency_scores(
        dmus, [0.5, 1, 1, 1, 1], model_solution, data)

    utils_for_tests.check_lambda_variables('A', 'B', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('B', 'B', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('C', 'C', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('E', 'E', 1, model_solution, data)
    utils_for_tests.check_lambda_variables('D', 'D', 1, model_solution, data)

    # The optimal weights of D and E are not unique and CBC can return
    # extreme values for them, so any valid optimal solution is accepted
    # instead of one particular weight vector (and VRS intercept).
    utils_for_tests.check_multiplier_solution_consistent(
        model_solution, data, dmus, 'output', vrs=True,
        free_categories={'q'})
