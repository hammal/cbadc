from typing import Dict, List, Set
from .. import (
    SPICE_VALUE,
    DeviceModel,
    Port,
    Terminal,
    SubCircuitElement,
    _template_env,
)
from ..components.voltage_buffer import VoltageBuffer
from ..components.ota import OTA
from ..components.passives import Resistor, Capacitor
import numpy as np


class OpAmp(SubCircuitElement):
    def __init__(
        self,
        instance_name: str,
        sub_ckt_name: str,
        GBWP: float,
        DC_gain: float,
        comments: List[str] = [],
        C: float = 10e-15,
    ):
        self.DC_gain = DC_gain
        self.GBWP = GBWP
        self.C = C
        self.gm = 2 * np.pi * GBWP * C
        self.R = DC_gain / (self.gm)

        X = Terminal('X', hidden=True)

        super().__init__(
            instance_name,
            sub_ckt_name,
            [
                Terminal('VDD'),
                Terminal('VSS'),
                Terminal('VCM'),
                Terminal('IN_P'),
                Terminal('IN_N'),
                Terminal('OUT_P'),
                Terminal('OUT_N'),
            ],
            comments=comments,
        )

        self.Gota = OTA('Gota', 'ota', gm=self.gm)
        self.Ebuf_p = VoltageBuffer('Ebuf_p', 'voltage_buffer')
        self.Ebuf_n = VoltageBuffer('Ebuf_n', 'voltage_buffer')
        self.Rx = Resistor('Rx', self.R)
        self.Cx = Capacitor('Cx', self.C)

        self.connects(
            (self['VDD'], self.Gota['VDD']),
            (self['VSS'], self.Gota['VSS']),
            (self['VDD'], self.Ebuf_p['VDD']),
            (self['VSS'], self.Ebuf_p['VSS']),
            (self['VDD'], self.Ebuf_n['VDD']),
            (self['VSS'], self.Ebuf_n['VSS']),
            (self['IN_P'], self.Gota['IN_P']),
            (self['IN_N'], self.Gota['IN_N']),
            (self['OUT_P'], self.Ebuf_p['OUT_P']),
            (self['VCM'], self.Ebuf_p['OUT_N']),
            (self['OUT_N'], self.Ebuf_n['OUT_P']),
            (self['VCM'], self.Ebuf_n['OUT_N']),
            (X, self.Ebuf_p['IN_P']),
            (X, self.Ebuf_n['IN_N']),
            (X, self.Gota['OUT_P']),
            (self['VCM'], self.Ebuf_p['IN_N']),
            (self['VCM'], self.Ebuf_n['IN_P']),
            (self['VCM'], self.Gota['OUT_N']),
            (X, self.Rx[0]),
            (X, self.Cx[0]),
            (self['VCM'], self.Rx[1]),
            (self['VCM'], self.Cx[1]),
        )

        # # Connect VDD and VSS
        # self.connect(self.terminals[0], self.ota._terminals[0])
        # self.connect(self.terminals[1], self.ota._terminals[1])
        # self.connect(self.terminals[0], self.vb[0]._terminals[0])
        # self.connect(self.terminals[1], self.vb[0]._terminals[1])
        # self.connect(self.terminals[0], self.vb[1]._terminals[0])
        # self.connect(self.terminals[1], self.vb[1]._terminals[1])

        # # Connect input
        # self.connect(self.terminals[3], self.ota._terminals[2])
        # self.connect(self.terminals[4], self.ota._terminals[3])

        # # Connect output
        # self.connect(self.terminals[5], self.vb[0]._terminals[4])
        # self.connect(self.terminals[2], self.vb[0]._terminals[5])
        # self.connect(self.terminals[6], self.vb[1]._terminals[4])
        # self.connect(self.terminals[2], self.vb[1]._terminals[5])

        # # Connect OTA output to buffer input
        # self.connect(X, self.vb[0]._terminals[2])
        # self.connect(X, self.vb[1]._terminals[3])
        # self.connect(X, self.ota._terminals[4])

        # self.connect(self.terminals[2], self.vb[0]._terminals[3])
        # self.connect(self.terminals[2], self.vb[1]._terminals[2])
        # self.connect(self.terminals[2], self.ota._terminals[5])

        # # Connect buffer output to resistor and capacitor
        # self.connect(X, self.r._terminals[0])
        # self.connect(X, self.c._terminals[0])

        # self.connect(self.terminals[2], self.r._terminals[1])
        # self.connect(self.terminals[2], self.c._terminals[1])
