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

    def test_timeseries_default(self):
        """Test run_length_control function."""
        # Create the property instance from a property file
        traj_file = os.path.join("tests", "fixtures", "lmp_T293.15.log")
        self.assertTrue(os.path.isfile(traj_file))

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

        data = \
            np.array(data, dtype=np.float64).reshape((len(data), len(data[0])))

        temp = data[:, 5]
        press = data[:, 9]

        global start, stop
        start = 0
        stop = 0

        def temp_get_trajectory(step):
            global start, stop
            if not isinstance(step, int):
                msg = "step number should be an `int`."
                raise CVGError(msg)
            start = stop
            temp_length = temp.shape[0]
            if temp_length < start + step:
                step = temp_length - start
            stop += step
            print('step={}, start={}, stop={}'.format(step, start, stop))
            return temp[start:stop]

        msg = cvg.run_length_control(
            get_trajectory=temp_get_trajectory,
            n_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=100000,
            sample_size=None,
            relative_accuracy=0.01,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='subsample',
            heidel_welch_number_points=50,
            fft=True,
            test_size=None,
            train_size=None,
            batch_size=5,
            scale='translate_scale',
            with_centering=False,
            with_scaling=False,
            ignore_end_batch=None,
            si='statistical_inefficiency',
            nskip=1,
            minimum_correlation_time=None,
            ignore_end=None,
            fp="return",
            fp_format='edn')

        self.assertTrue(msg)

        kim_obj = kim_edn.loads(msg)

        self.assertTrue(kim_obj["converged"])
        self.assertTrue(kim_obj["effective_sample_size"] < 30)
        equilibration_step = kim_obj["equilibration_step"]

        start = 0
        stop = 0

        msg = cvg.run_length_control(
            get_trajectory=temp_get_trajectory,
            n_variables=1,
            initial_run_length=1000,
            run_length_factor=1.5,
            maximum_run_length=100000,
            sample_size=30,
            relative_accuracy=0.01,
            confidence_coefficient=0.95,
            confidence_interval_approximation_method='subsample',
            heidel_welch_number_points=50,
            fft=True,
            test_size=None,
            train_size=None,
            batch_size=5,
            scale='translate_scale',
            with_centering=False,
            with_scaling=False,
            ignore_end_batch=None,
            si='statistical_inefficiency',
            nskip=1,
            minimum_correlation_time=None,
            ignore_end=None,
            fp="return",
            fp_format='json')

        self.assertTrue(msg)

        json_obj = json.loads(msg)

        self.assertTrue(json_obj["converged"])
        self.assertTrue(json_obj["effective_sample_size"] >= 30)
        self.assertTrue(equilibration_step == json_obj["equilibration_step"])
