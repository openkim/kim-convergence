"""Test timeseries module."""
import json
import kim_edn
import numpy as np
import os
import unittest

try:
    import kim_convergence as cr
except:
    raise Exception('Failed to import `kim-convergence` utility module')

from kim_convergence import CRError

start = 0
stop = 0


class TimeseriesModule(unittest.TestCase):
    """Test timeseries module components."""

    # Read the trajectory file
    traj_file = os.path.join('tests', 'fixtures', 'lmp_T293.15.log')
    with open(traj_file, 'r') as f_in:
        lines = f_in.readlines()
    header = None
    data = []
    for line in lines:
        words = line.strip().split()
        if len(words) < 1:
            continue
        if words[0] == 'Step':
            if header is None:
                header = line.strip().split()
            continue
        if len(words) > 10:
            try:
                step = float(words[0])
            except:
                continue
            if step > len(data):
                data.append(words)
    data = np.array(data, dtype=np.float64).reshape((len(data), len(data[0])))

    def test_timeseries_default(self):
        """Test run_length_control function."""
        global start, stop

        temp = self.data[:, 5]
        temp_size = temp.size
        press = self.data[:, 9]
        press_size = press.size
        volume = self.data[:, 7]

        print('\n')

        start = 0
        stop = 0

        def temp_get_trajectory(step):
            global start, stop
            start = stop
            if temp_size < start + step:
                step = temp_size - start
            stop += step
            # print(f'{step=}, {start=}, {stop=}')
            return temp[start:stop]

        msg = cr.run_length_control(
            get_trajectory=temp_get_trajectory,
            number_of_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=temp_size,
            maximum_equilibration_step=temp_size // 2,
            minimum_number_of_independent_samples=None,
            relative_accuracy=0.01,
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='uncorrelated_sample',
            fp='return',
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)

        subsample_effective_sample_size = kim_obj['effective_sample_size']
        subsample_equilibration_step = kim_obj['equilibration_step']

        self.assertTrue(kim_obj['converged'])
        self.assertTrue(subsample_effective_sample_size < 35)
        self.assertTrue(subsample_equilibration_step < 1300)

        start = 0
        stop = 0

        msg = cr.run_length_control(
            get_trajectory=temp_get_trajectory,
            number_of_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=temp_size,
            maximum_equilibration_step=temp_size // 2,
            minimum_number_of_independent_samples=30,
            relative_accuracy=0.01,
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='uncorrelated_sample',
            fp='return',
            fp_format='json')

        json_obj = json.loads(msg)

        self.assertTrue(json_obj['converged'])
        self.assertTrue(json_obj['effective_sample_size'] >= 30)
        self.assertTrue(subsample_equilibration_step ==
                        json_obj['equilibration_step'])

        # heidel_welch
        start = 0
        stop = 0

        msg = cr.run_length_control(
            get_trajectory=temp_get_trajectory,
            number_of_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=temp_size,
            maximum_equilibration_step=temp_size // 2,
            minimum_number_of_independent_samples=None,
            relative_accuracy=0.01,
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='heidel_welch',
            fp='return',
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)

        heidel_welch_effective_sample_size = kim_obj['effective_sample_size']
        heidel_welch_equilibration_step = kim_obj['equilibration_step']

        self.assertTrue(kim_obj['converged'])
        self.assertTrue(heidel_welch_effective_sample_size ==
                        subsample_effective_sample_size)
        self.assertTrue(heidel_welch_equilibration_step ==
                        subsample_equilibration_step)

        start = 0
        stop = 0

        def press_get_trajectory(step):
            global start, stop
            start = stop
            if press_size < start + step:
                step = press_size - start
            stop += step
            return press[start:stop]

        msg = cr.run_length_control(
            get_trajectory=press_get_trajectory,
            number_of_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=press_size,
            maximum_equilibration_step=press_size // 2,
            minimum_number_of_independent_samples=None,
            relative_accuracy=0.10,
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='uncorrelated_sample',
            fp='return',
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)
        self.assertFalse(kim_obj['converged'])
        self.assertTrue(kim_obj['equilibration_detected'])
        self.assertTrue(kim_obj['effective_sample_size'] >= 300)

        start = 0
        stop = 0

        def temp_press_get_trajectory(step):
            global start, stop
            start = stop
            if press_size < start + step:
                step = press_size - start
            stop += step
            traj = np.concatenate((temp[start:stop],
                                   press[start:stop])).reshape((2, -1))
            # print(f'{step=}, {start=}, {stop=}')
            return traj

        msg = cr.run_length_control(
            get_trajectory=temp_press_get_trajectory,
            number_of_variables=2,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=press_size,
            maximum_equilibration_step=press_size // 2,
            minimum_number_of_independent_samples=None,
            relative_accuracy=0.10,
            absolute_accuracy=None,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='uncorrelated_sample',
            fp='return',
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)
        self.assertFalse(kim_obj['converged'])
        self.assertTrue(kim_obj['equilibration_detected'])
        self.assertTrue(kim_obj['0']['effective_sample_size'] >= 150)
        self.assertTrue(kim_obj['1']['effective_sample_size'] >= 300)

        start = 0
        stop = 0

        def temp_press_volume_get_trajectory(step):
            global start, stop
            start = stop
            if press_size < start + step:
                step = press_size - start
            stop += step
            traj = np.concatenate((temp[start:stop],
                                   press[start:stop],
                                   volume[start:stop])).reshape((3, -1))
            # print(f'{step=}, {start=}, {stop=}')
            return traj

        msg = cr.run_length_control(
            get_trajectory=temp_press_volume_get_trajectory,
            number_of_variables=3,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=press_size,
            maximum_equilibration_step=press_size // 2,
            minimum_number_of_independent_samples=None,
            relative_accuracy=[0.10, None, 0.1],
            absolute_accuracy=[None, 0.1, None],
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='uncorrelated_sample',
            fp='return',
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)

        self.assertFalse(kim_obj['converged'])
        self.assertTrue(kim_obj['equilibration_detected'])
        self.assertTrue(kim_obj['0']['effective_sample_size'] >= 150)
        self.assertTrue(kim_obj['1']['effective_sample_size'] >= 300)
        self.assertTrue(kim_obj['2']['effective_sample_size'] >= 250)
