from dataclasses import dataclass
import logging
from typing import Generator, List
from .simulator import FullSimulator
from .digital_estimator import BatchEstimator
from .utilities import show_status
from .utilities import pickle_dump, pickle_load
import numpy as np
import pandas as pd
import os
from scipy.integrate._ivp.ivp import OdeResult
from .simulation_event import SimulationEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_default_batch_size = 1 << 12


@dataclass()
class Recordings:
    simulator: FullSimulator
    estimator: BatchEstimator
    discrete_filenames: List[str]
    continuous_filenames: List[str]


class Recorder:

    simulator: FullSimulator
    estimator: BatchEstimator

    def __init__(
        self,
        filename: str,
        simulator: FullSimulator,
        estimator: BatchEstimator,
        batch_size=_default_batch_size,
        reset_file=False,
    ):
        self.filename = filename
        if reset_file and os.path.exists(self.filename):
            os.remove(self.filename)
        else:
            logger.info(f"no file: {self.filename} to delete.")

        if (batch_size % 2) != 0:
            raise Exception("Batch size must be multiple of 2.")
        self.batch_size = batch_size

        self.simulator = simulator
        self.estimator = estimator
        self.estimator(self.simulator)

        self._index = 1

        self._input_vector = np.zeros(self.simulator.analog_system.L)

        self._discrete_filenames = []
        self._continuous_filenames = []

    def _inputs_at_time(self, t):
        for _l in range(self.simulator.analog_system.L):
            self._input_vector[_l] = self.simulator.input_signals[_l].evaluate(t)
        return self._input_vector

    def _states_at_time(self, t, res):
        if t in res.t:
            return res.y[:, res.t == t].flatten()
        for event_index, t_event in enumerate(res.t_events):
            if t in t_event:
                return res.y_events[event_index][t_event == t].flatten()
        raise Exception("No such state in simulation?")

    def _discrete_time_observation(self, t, u_hat):
        # Evaluate true input signal values at current time
        return np.hstack(
            (
                np.array([t]),
                self.simulator.digital_control.control_signal(),
                self._inputs_at_time(t),
                u_hat,
            )
        )

    def _continuous_time_observation(self, t, res):
        # Evaluate true input signal values at current time
        x = self._states_at_time(t, res)
        u = self._inputs_at_time(t)
        s = self.simulator.digital_control.control_contribution(t)
        return np.hstack(
            (
                np.array([t]),
                x,
                s,
                self.simulator.digital_control.control_signal(),
                self.simulator.analog_system.control_observation(x, u, s),
                u,
            )
        )

    def flush(self, discrete: pd.DataFrame, continuous: pd.DataFrame, index: int = 0):
        # discrete.to_hdf(self.filename, 'discrete', 'a', append=True)
        # continuous.to_hdf(self.filename, 'continuous', 'a', append=True)
        self._discrete_filenames.append(f"{self.filename}_discrete_{index}.data")
        self._continuous_filenames.append(f"{self.filename}_continuous_{index}.data")
        discrete.to_feather(self._discrete_filenames[-1])
        continuous.to_feather(self._continuous_filenames[-1])
        filename = self.filename
        pickle_dump(
            Recordings(
                discrete_filenames=self._discrete_filenames,
                continuous_filenames=self._continuous_filenames,
                simulator=self.simulator,
                estimator=self.estimator,
            ),
            filename,
        )

    def save(self, num_samples=_default_batch_size):
        logger.info(f"recording {num_samples} samples")
        discrete_data = []
        continuous_data = []
        for index in show_status(range(num_samples)):
            # Get next estimate
            u_hat = next(self.estimator)

            # Store signals in data buffer
            dt_observation = self._discrete_time_observation(self.simulator.t, u_hat)
            data_temp = {
                't': dt_observation[0],
            }
            offset = 1
            for s_index in range(self.simulator.analog_system.M):
                data_temp[f"s_k_{s_index + 1}"] = dt_observation[s_index + offset]
            offset += self.simulator.analog_system.M
            for u_index in range(self.simulator.analog_system.L):
                data_temp[f"u_{u_index + 1}"] = dt_observation[u_index + offset]
            offset += self.simulator.analog_system.L
            for u_index in range(self.simulator.analog_system.L):
                data_temp[f"u_hat_{u_index + 1}"] = dt_observation[u_index + offset]

            discrete_data.append(
                pd.DataFrame(
                    data_temp,
                    index=[index],
                )
            )

            sim_res: OdeResult = self.simulator.res

            c_times = []
            event_buffer = []
            data = []

            # Collect all events
            for event_index, event in enumerate(sim_res.event_list):
                for t in sim_res.t_events[event_index]:
                    if t not in c_times:
                        data.append(self._continuous_time_observation(t, sim_res))
                        c_times.append(t)
                        event_buffer.append(event)

            # Collect all samples
            for t in sim_res.t[1:]:
                if t not in c_times:
                    data.append(self._continuous_time_observation(t, sim_res))
                    c_times.append(t)
                    event_buffer.append(SimulationEvent('sample'))

            # Sort with respect to time
            sort_index = np.argsort(c_times)
            data = np.array(data)
            event_buffer = np.array(event_buffer)

            df_data = {"t": data[:, 0]}
            offset = 1
            for x_index in range(self.simulator.analog_system.N):
                df_data[f"x_{x_index + 1}"] = data[sort_index, x_index + offset]
            offset += self.simulator.analog_system.N
            for s_index in range(self.simulator.analog_system.M):
                df_data[f"s_t_{s_index + 1}"] = data[sort_index, s_index + offset]
            offset += self.simulator.analog_system.M
            for s_index in range(self.simulator.analog_system.M):
                df_data[f"s_k_{s_index + 1}"] = data[sort_index, s_index + offset]
            offset += self.simulator.analog_system.M
            for s_index in range(self.simulator.analog_system.M_tilde):
                df_data[f"s_tilde_{s_index + 1}"] = data[sort_index, s_index + offset]
            offset += self.simulator.analog_system.M_tilde
            for u_index in range(self.simulator.analog_system.L):
                df_data[f"u_{u_index + 1}"] = data[sort_index, u_index + offset]
            offset += self.simulator.analog_system.L
            df_data['event'] = np.array([e.name for e in event_buffer])[sort_index]

            continuous_data.append(pd.DataFrame(df_data))

            if (index % self.batch_size) == 0 and index > 0:
                self.flush(
                    pd.concat(discrete_data, ignore_index=True),
                    pd.concat(continuous_data, ignore_index=True),
                    self._index,
                )
                discrete_data = []
                continuous_data = []
                self._index += 1
        if len(discrete_data) > 0 or len(continuous_data) > 0:
            self.flush(
                pd.concat(discrete_data, ignore_index=True),
                pd.concat(continuous_data, ignore_index=True),
                self._index,
            )


class Playback:
    def __init__(self, filename: str):
        self.filename = filename
        pickled_items: Recordings = pickle_load(self.filename)
        self._discrete_filenames = pickled_items.discrete_filenames
        self._continuous_filenames = pickled_items.continuous_filenames
        self.estimator = pickled_items.simulator
        self.simulator = pickled_items.estimator

    def _get_pandas_df(self, list_of_filenames):
        dfs = []
        for filename in list_of_filenames:
            dfs.append(pd.read_feather(filename))
        return pd.concat(dfs, ignore_index=True)

    def _get_discrete_time_pandas_df(self):
        return self._get_pandas_df(self._discrete_filenames)

    def _get_continuous_time_pandas_df(self):
        return self._get_pandas_df(self._continuous_filenames)
