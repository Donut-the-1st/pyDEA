''' This module contains MultiplierModelBase class that
    implements basic functionality of the multiplier model.
'''
import logging

import pulp

from pyDEA.core.models.model_base import ModelBase
from pyDEA.core.models.multiplier_solution_check import certified_dual_bound
from pyDEA.core.models.multiplier_solution_check import check_primal
from pyDEA.core.models.multiplier_solution_check import (
    select_bounded_solution)
from pyDEA.core.utils.dea_utils import is_efficient, ZERO_TOLERANCE

# the root logger is used, as everywhere in pyDEA: get_logger() configures
# logging with fileConfig, which disables loggers that already exist
logger = logging.getLogger()

# maximum deviation from 1 of the efficiency score computed from the
# objective value of a DMU whose own lambda variable is positive
EFFICIENCY_CONSISTENCY_TOLERANCE = 1e-6


class MultiplierModelBase(ModelBase):
    ''' This class is a base class for different multiplier models.
        It implements general structure of all multiplier models.
        Concrete variations of multiplier models are passed in the
        class constructor.

        Attributes:
            tolerance (double): lower bound of the variables of multilier
                model.
            _concrete_model: concrete implementation of the multiplier
                model.
            _input_variables (dict of str to pulp.LpVariable): dictionary of
                pulp variables than maps variable names to pulp variables
                corresponding to input categories.
            _output_variables (dict of str to pulp.LpVariable): dictionary of
                pulp variables than maps variable names to pulp variables
                corresponding to output categories.
            _dmu_constraint_names (dict of str to str): dictionary that maps
                constraint names to DMU code.

        Args:
            input_data (InputData): object that stores all data of
                a DEA instance.
            tolerance (double): lower bound of the variables of multilier
                model.
            concrete_model: concrete implementation of the multiplier
                model.
    '''
    def __init__(self, input_data, tolerance, concrete_model):
        super(MultiplierModelBase, self).__init__(input_data)
        self.tolerance = tolerance
        self._concrete_model = concrete_model
        self._input_variables = dict()
        self._output_variables = dict()
        self._dmu_constraint_names = dict()

    def _create_lp(self):
        ''' Creates initial LP model.
        '''
        assert len(self.input_data.DMU_codes) != 0
        # create LP for the first DMU - it is the easiest way to
        # adapt current code
        for elem in self.input_data.DMU_codes:
            dmu_code = elem
            break

        self.lp_model = pulp.LpProblem(
            'Multiplier model: {orientation}-oriented'.format(
            orientation=self._concrete_model.get_orientation()),
            self._concrete_model.get_objective_type())

        self._output_variables = pulp.LpVariable.dicts(
            'mu', self.input_data.output_categories, self.tolerance, None,
            pulp.LpContinuous)
        self._input_variables = pulp.LpVariable.dicts(
            'eta', self.input_data.input_categories, self.tolerance, None,
            pulp.LpContinuous)

        self.lp_model += (self._concrete_model.get_objective_function(
            self.input_data, dmu_code, self._input_variables,
            self._output_variables),
            'Efficiency score or inverse of efficiency score')

        self._dmu_constraint_names.clear()
        for dmu in self.input_data.DMU_codes:
            output_sum = pulp.lpSum([self.input_data.coefficients[
                dmu, category] * self._output_variables[category]
                for category in self.input_data.output_categories])
            input_sum = pulp.lpSum([self.input_data.coefficients[
                dmu, category] * self._input_variables[category]
                for category in self.input_data.input_categories])
            name = 'DMU_constraint_{count}'.format(count=dmu)
            self.lp_model += (output_sum - input_sum <= 0, name)
            self._dmu_constraint_names[name] = dmu

        self.lp_model += (self._concrete_model.get_equality_constraint(
            self.input_data, dmu_code, self._input_variables,
            self._output_variables), 'equality_constraint')

    def _update_lp(self, dmu_code):
        ''' Updates existing linear program with coefficients corresponding
            to a given DMU.

            Args:
                dmu_code (str): DMU code.
        '''
        self._concrete_model.update_objective(self.input_data, dmu_code,
                                              self._input_variables,
                                              self._output_variables,
                                              self.lp_model)
        self._concrete_model.update_equality_constraint(self.input_data,
                                                        dmu_code,
                                                        self._input_variables,
                                                        self._output_variables,
                                                        self.lp_model)

    def _fill_solution(self, dmu_code, model_solution):
        ''' Fills given solution with data calculated for one DMU.

            Args:
                dmu_code (str): DMU code for which the LP was solved.
                model_solution (Solution): object where solution for one DMU
                    will be written.
        '''
        model_solution.orientation = self._concrete_model.get_orientation()
        model_solution.add_lp_status(dmu_code, self.lp_model.status)
        if self.lp_model.status == pulp.LpStatusOptimal:
            # dual values must be read before _ensure_valid_weights, which
            # may solve a second LP that shares variables with this one
            lambda_variables = dict()
            for dmu_constraint, dmu in self._dmu_constraint_names.items():
                constraint = self.lp_model.constraints[dmu_constraint]
                if constraint.pi is None:
                    constraint.pi = 0
                if abs(constraint.pi) > ZERO_TOLERANCE:
                    lambda_variables[dmu] = self._concrete_model.process_dual_value(
                        constraint.pi)
            lambda_var = lambda_variables.get(dmu_code, 0)

            obj_value = self._ensure_valid_weights(dmu_code, model_solution)

            for input_category in self.input_data.input_categories:
                model_solution.add_input_dual(
                    dmu_code, input_category,
                    self._input_variables[input_category].varValue)

            for output_category in self.input_data.output_categories:
                model_solution.add_output_dual(
                    dmu_code, output_category,
                    self._output_variables[output_category].varValue)

            self._check_efficiency_consistency(
                dmu_code, self._get_raw_efficiency_score(obj_value),
                lambda_var, model_solution)
            model_solution.add_efficiency_score(
                dmu_code, self._get_efficiency_score(lambda_var, obj_value))
            model_solution.add_lambda_variables(dmu_code, lambda_variables)

    def _ensure_valid_weights(self, dmu_code, model_solution):
        ''' Checks the weights returned by the solver for the current DMU
            and, if they are inconsistent or numerically unreliable,
            replaces them with another optimal solution (see
            multiplier_solution_check). Problems that cannot be resolved
            are recorded as numerical warnings in model_solution and logged.

            Args:
                dmu_code (str): DMU code for which the LP was solved.
                model_solution (Solution): solution where warnings are
                    recorded.

            Returns:
                double or None: objective value that must be used for the
                    efficiency score, or None if the objective value
                    computed from the variable values must be used.
        '''
        z_star, scale, dual_ok = certified_dual_bound(self.lp_model)
        if not dual_ok:
            is_ok, info = check_primal(self.lp_model)
            self._add_numerical_warning(
                dmu_code, model_solution,
                'the dual solution returned by the solver failed '
                'verification, so lambda values may be unreliable')
            if not is_ok:
                self._add_numerical_warning(
                    dmu_code, model_solution,
                    'weights returned by the solver are inconsistent '
                    '(worst violation {0:.3g}, cancellation ratio {1:.3g})'
                    .format(info['violation'], info['cancellation']))
            return None

        is_ok, info = check_primal(self.lp_model, z_star)
        if is_ok:
            return None

        selected = select_bounded_solution(
            self.lp_model, z_star, scale, self._get_variable_scales())
        if selected:
            logger.info(
                'Multiplier model, DMU %s: weights returned by the solver '
                'were inconsistent or unreliable (objective gap %.3g, worst '
                'violation %.3g, cancellation ratio %.3g); another optimal '
                'solution was selected.', self._get_dmu_name(dmu_code),
                info['gap'], info['violation'], info['cancellation'])
        else:
            self._add_numerical_warning(
                dmu_code, model_solution,
                'weights returned by the solver are inconsistent (objective '
                'gap {0:.3g}, worst violation {1:.3g}, cancellation ratio '
                '{2:.3g}) and no consistent optimal solution could be '
                'selected'.format(info['gap'], info['violation'],
                                  info['cancellation']))
        return z_star

    def _get_variable_scales(self):
        ''' Returns penalties used for selection of an optimal solution:
            the mean absolute value of the data of the corresponding
            category, or 1 if it is zero. Other variables (for example the
            VRS variable) get penalty 1.

            Returns:
                dict of pulp.LpVariable to double: penalties.
        '''
        scales = dict()
        categories = (list(self._input_variables.items()) +
                      list(self._output_variables.items()))
        for category, var in categories:
            values = [abs(self.input_data.coefficients[dmu, category])
                      for dmu in self.input_data.DMU_codes]
            mean_value = sum(values) / len(values) if values else 0
            scales[var] = mean_value if mean_value > 0 else 1.0
        return scales

    def _get_dmu_name(self, dmu_code):
        ''' Returns user name of a given DMU, or its code if the name is not
            known.
        '''
        try:
            return self.input_data.get_dmu_user_name(dmu_code)
        except (KeyError, ValueError, AttributeError):
            return dmu_code

    def _add_numerical_warning(self, dmu_code, model_solution, message):
        ''' Records and logs a numerical warning for a given DMU.
        '''
        model_solution.add_numerical_warning(dmu_code, message)
        logger.warning('Multiplier model, DMU %s: %s',
                       self._get_dmu_name(dmu_code), message)

    def _check_efficiency_consistency(self, dmu_code, raw_score,
                                      lambda_variable, model_solution):
        ''' Records a numerical warning if the efficiency score computed
            from the objective value contradicts the dual solution, i.e. if
            the efficiency score would only be set to 1 by is_efficient
            because of a positive lambda variable of the DMU itself. By
            complementary slackness the objective value must correspond to
            efficiency 1 in this case. Scores greater than 1 are also
            impossible while the DMU's own constraint is present. The check
            is skipped for super efficiency models, where this constraint is
            removed.

            Args:
                dmu_code (str): DMU code.
                raw_score (double): efficiency score computed from the
                    objective value.
                lambda_variable (double): lambda variable of the DMU itself.
                model_solution (Solution): solution where warnings are
                    recorded.
        '''
        if dmu_code not in self._dmu_constraint_names.values():
            return
        deviation = abs(raw_score - 1)
        if ((lambda_variable > ZERO_TOLERANCE and
             deviation > EFFICIENCY_CONSISTENCY_TOLERANCE) or
                raw_score > 1 + EFFICIENCY_CONSISTENCY_TOLERANCE):
            self._add_numerical_warning(
                dmu_code, model_solution,
                'efficiency score computed from the objective value ({0}) is '
                'inconsistent with the dual solution and was set to 1'
                .format(raw_score))

    def _get_raw_efficiency_score(self, obj_value=None):
        ''' Returns efficiency score computed from the objective value,
            before it is adjusted by is_efficient.

            Args:
                obj_value (double, optional): objective value. If None, it
                    is computed from the variable values.

            Returns:
                double: efficiency score.
        '''
        if obj_value is None:
            obj_value = pulp.value(self.lp_model.objective)
        # None is possible when objective function is zero; it must be
        # handled before process_obj_var, which cannot compare None
        if obj_value is None:
            return 0
        return self._concrete_model.process_obj_var(obj_value)

    def _get_efficiency_score(self, lambda_variable, obj_value=None):
        ''' Returns efficiency score based on a given lambda variable.

            Args:
                lambda_variable (double): value of the lambda variable.
                obj_value (double, optional): objective value. If None, it
                    is computed from the variable values.

            Returns:
                double: efficiency spyDEA.core.
        '''
        eff_score = self._get_raw_efficiency_score(obj_value)
        if is_efficient(eff_score, lambda_variable):
            eff_score = 1
        return eff_score
