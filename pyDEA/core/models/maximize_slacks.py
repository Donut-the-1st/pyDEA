''' This module contains the class responsible for implementing a
    two-phase model.
'''
import pulp

from pyDEA.core.models.envelopment_model_base import EnvelopmentModelBase
from pyDEA.core.utils.dea_utils import get_lp_solver, ZERO_TOLERANCE

# relative tolerance used to fix the phase-1 objective variable in phase 2
PHASE_ONE_OBJECTIVE_TOLERANCE = 1e-7


class MaximizeSlacksModel(EnvelopmentModelBase):
    ''' Implements a two-phase model.

        Attributes:
            strongly_disposal_input_categories (set of str): set
                of strongly disposal input categories.
            strongly_disposal_output_categories (set of str): set
                of strongly disposal output categories.
            model (EnvelopmentModelBase): envelopment model used in
                two-phase model.
            second_solution (Solution): solution obtained after second
                phase.
            _input_slack_vars (dict of str to pulp.LpVariable): slack
                variables of strongly disposable input categories.
            _output_slack_vars (dict of str to pulp.LpVariable): slack
                variables of strongly disposable output categories.

        Args:
            model (EnvelopmentModelBase): envelopment model.
            weakly_disposable_categories (set of str, optional): set
                of weakly disposal categories. Defaults to None.
    '''
    def __init__(self, model, weakly_disposable_categories=None):
        if weakly_disposable_categories is None:
            weakly_disposable_categories = set()
        input_categories = model.input_data.input_categories
        self.strongly_disposal_input_categories = input_categories.difference(
            weakly_disposable_categories)
        output_categories = model.input_data.output_categories
        self.strongly_disposal_output_categories = output_categories.difference(
            weakly_disposable_categories)
        self.model = model
        self.second_solution = None
        self._input_slack_vars = {}
        self._output_slack_vars = {}

    def __getattr__(self, name):
        return getattr(self.model, name)

    def run_for_one_DMU(self, dmu_code, model_solution):
        ''' Solves phase 1 for a given DMU, then phase 2: with the phase-1
            objective variable (theta or phi) fixed, maximises the sum of
            slacks of all strongly disposable categories.

            Phase 2 is built from a copy of the phase-1 LP after it has
            been updated for this DMU, so every decorator (VRS, weight
            restrictions, super efficiency) is already applied to it.
            The second solution gets the phase-1 efficiency score and duals,
            and the phase-2 LP status and lambda variables.

            Note:
                With weight restrictions, their extra variables stay in the
                copied constraints and only restrict feasibility in phase 2.
        '''
        self.model.run_for_one_DMU(dmu_code, model_solution)
        assert self.second_solution is not None
        # copies phase-1 status, efficiency, lambdas and duals (including
        # those added by decorators) into the second solution
        self.model._fill_solution(dmu_code, self.second_solution)
        if self.model.lp_model.status != pulp.LpStatusOptimal:
            return

        lp_phase_two = self._create_phase_two_lp()
        lp_phase_two.solve(get_lp_solver())
        self.second_solution.add_lp_status(dmu_code, lp_phase_two.status)
        if lp_phase_two.status == pulp.LpStatusOptimal:
            # LpProblem.deepcopy shares variables, so the lambda variables
            # now hold phase-2 values
            lambda_variables = dict()
            for dmu in self.input_data.DMU_codes:
                var = self.model._variables.get(dmu, None)
                if (var is not None and var.varValue is not None and
                        abs(var.varValue) > ZERO_TOLERANCE):
                    lambda_variables[dmu] = var.varValue
            self.second_solution.add_lambda_variables(dmu_code,
                                                      lambda_variables)

    def _create_phase_two_lp(self):
        ''' Creates the phase-2 LP from the phase-1 LP that has just been
            solved for the current DMU.

            Returns:
                pulp.LpProblem: phase-2 LP.
        '''
        lp_phase_two = self.model.lp_model.deepcopy()
        lp_phase_two.sense = pulp.LpMaximize
        lp_phase_two.objective = (
            pulp.lpSum(list(self._input_slack_vars.values())) +
            pulp.lpSum(list(self._output_slack_vars.values())))

        # constraints of both inputs and outputs have the form expr >= 0,
        # the slack is the surplus: expr - slack = 0
        for category, slack in list(self._input_slack_vars.items()) + list(
                self._output_slack_vars.items()):
            constraint = lp_phase_two.constraints[
                self.model._constraints[category]]
            constraint.expr.addterm(slack, -1)
            constraint.sense = pulp.LpConstraintEQ

        # Fix the efficiency score (or its inverse) at its phase-1 value.
        # Constraints are used instead of variable bounds, because variables
        # are shared with the phase-1 LP. CBC reports values with about 8
        # significant digits, so fixing the rounded value exactly can make
        # phase 2 infeasible (e.g. 11/9 reported as 1.2222222); a narrow
        # band around it is allowed instead.
        obj_var = self.model._variables['obj_var']
        obj_value = obj_var.varValue
        band = PHASE_ONE_OBJECTIVE_TOLERANCE * max(1.0, abs(obj_value))
        lp_phase_two += (obj_var >= obj_value - band,
                         'phase_one_objective_lower')
        lp_phase_two += (obj_var <= obj_value + band,
                         'phase_one_objective_upper')
        return lp_phase_two

    # we need to explicitly redirect all calls to model since this class
    # inherits from EnvelopmentModelBase and all methods are defined there.
    # Some of them might be redefined by children classes. To ensure children
    # redefined methods are called, we have to explicitly redirect calls to
    # stored model.
    def _create_lp(self):
        ''' See base class.
        '''
        self.model._create_lp()
        self._input_slack_vars = pulp.LpVariable.dicts(
            'input_slack', self.strongly_disposal_input_categories,
            0, None, pulp.LpContinuous)
        self._output_slack_vars = pulp.LpVariable.dicts(
            'output_slack', self.strongly_disposal_output_categories,
            0, None, pulp.LpContinuous)

    def _update_lp(self, dmu_code):
        ''' See base class.
        '''
        self.model._update_lp(dmu_code)

    def _add_constraints_for_outputs(self, variables, dmu_code,
                                     obj_variable):
        ''' See base class.
        '''
        self.model._add_constraints_for_outputs(variables, dmu_code,
                                                obj_variable)

    def _add_constraints_for_inputs(self, variables,
                                    dmu_code, obj_variable):
        ''' See base class.
        '''
        self.model._add_constraints_for_inputs(variables, dmu_code,
                                               obj_variable)

    def _process_duals(self, dmu_code, categories, func):
        ''' See base class.
        '''
        self.model._process_duals(dmu_code, categories, func)

    def _fill_solution(self, dmu_code, model_solution):
        ''' See base class.
        '''
        self.model._fill_solution(dmu_code, model_solution)

    def _create_solution(self):
        ''' See base class.
        '''
        self.second_solution = self.model._create_solution()
        return self.model._create_solution()
