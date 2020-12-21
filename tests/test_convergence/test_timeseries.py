"""Test timeseries module."""
import os
import json
import unittest

import numpy as np
import kim_edn

try:
    import convergence as cvg
except:
    raise Exception('Failed to import `convergence` utility module')

from convergence import CVGError

start = 0
stop = 0


class TimeseriesModule(unittest.TestCase):
    """Test timeseries module components."""

    # Read the trajectory file
    traj_file = os.path.join("tests", "fixtures", "lmp_T293.15.log")
    with open(traj_file, "r") as f_in:
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

        start = 0
        stop = 0

        def temp_get_trajectory(step):
            global start, stop
            start = stop
            if temp_size < start + step:
                step = temp_size - start
            stop += step
            # print('step={}, start={}, stop={}'.format(step, start, stop))
            return temp[start:stop]

        msg = cvg.run_length_control(
            get_trajectory=temp_get_trajectory,
            n_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=temp_size,
            sample_size=None,
            relative_accuracy=0.01,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='subsample',
            fp="return",
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)
        subsample_effective_sample_size = kim_obj["effective_sample_size"]
        subsample_equilibration_step = kim_obj["equilibration_step"]

        self.assertTrue(kim_obj["converged"])
        self.assertTrue(subsample_effective_sample_size < 30)

        start = 0
        stop = 0

        msg = cvg.run_length_control(
            get_trajectory=temp_get_trajectory,
            n_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=temp_size,
            sample_size=30,
            relative_accuracy=0.01,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='subsample',
            fp="return",
            fp_format='json')

        json_obj = json.loads(msg)

        self.assertTrue(json_obj["converged"])
        self.assertTrue(json_obj["effective_sample_size"] >= 30)
        self.assertTrue(subsample_equilibration_step ==
                        json_obj["equilibration_step"])

        # heidel_welch
        start = 0
        stop = 0

        msg = cvg.run_length_control(
            get_trajectory=temp_get_trajectory,
            n_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=temp_size,
            sample_size=None,
            relative_accuracy=0.01,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='heidel_welch',
            fp="return",
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)

        heidel_welch_effective_sample_size = kim_obj["effective_sample_size"]
        heidel_welch_equilibration_step = kim_obj["equilibration_step"]

        self.assertTrue(kim_obj["converged"])
        self.assertTrue(heidel_welch_effective_sample_size <
                        subsample_effective_sample_size)
        self.assertTrue(heidel_welch_equilibration_step ==
                        subsample_equilibration_step)

        press = self.data[:, 9]
        press_size = press.size

        start = 0
        stop = 0

        def press_get_trajectory(step):
            global start, stop
            start = stop
            if press_size < start + step:
                step = press_size - start
            stop += step
            return press[start:stop]

        msg = cvg.run_length_control(
            get_trajectory=press_get_trajectory,
            n_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=press_size,
            sample_size=None,
            relative_accuracy=0.10,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='subsample',
            fp="return",
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)
        self.assertFalse(kim_obj["converged"])

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
            # print('step={}, start={}, stop={}'.format(step, start, stop))
            return traj

        msg = cvg.run_length_control(
            get_trajectory=temp_press_get_trajectory,
            n_variables=2,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=press_size,
            sample_size=None,
            relative_accuracy=0.10,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='subsample',
            fp="return",
            fp_format='edn')

        kim_obj = kim_edn.loads(msg)
        self.assertFalse(kim_obj["converged"])
