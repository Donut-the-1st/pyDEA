from types import SimpleNamespace

from pyDEA.core.utils.run_routine import (
    derive_returns_to_scale_classification,
)


class FakeSolution:
    def __init__(self, names, scores, lambda_sums):
        codes = list(names)
        self._input_data = SimpleNamespace(
            DMU_codes_in_added_order=codes,
            DMU_code_to_user_name=names,
        )
        self.efficiency_scores = scores
        self._lambda_sums = lambda_sums

    def get_lambda_variables(self, dmu_code):
        return {'lambda': self._lambda_sums[dmu_code]}


def test_derive_returns_to_scale_classification():
    names = {
        'equal': 'Equal',
        'irs': 'IRS',
        'drs': 'DRS',
    }
    crs_scores = {'equal': 0.8, 'irs': 0.7, 'drs': 0.6}
    vrs_scores = {'equal': 0.8 + 5e-11, 'irs': 0.8, 'drs': 0.9}
    crs_lambda_sums = {'equal': 0.5, 'irs': 0.75, 'drs': 1.25}
    vrs_lambda_sums = {dmu_code: 1 for dmu_code in names}

    param_strs = []
    solutions = []
    for orientation in ('input orientation', 'output orientation'):
        param_strs.extend((
            orientation + ', CRS',
            orientation + ', VRS',
        ))
        solutions.extend((
            FakeSolution(names, crs_scores, crs_lambda_sums),
            FakeSolution(names, vrs_scores, vrs_lambda_sums),
        ))

    derive_returns_to_scale_classification(param_strs, solutions)

    expected = {'equal': 'CRS', 'irs': 'IRS', 'drs': 'DRS'}
    for solution in solutions:
        assert solution.return_to_scale == expected

