# import h5py
from mpi4py import MPI
import numpy as np
import cbadc
import itertools
from cbadc import delsig as ds
from cbadc import AnalogFrontend, GmC, Sinusoidal
import logging
import time
import csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

Cint = np.geomspace(10e-15, 1e-12, 3)
Ro = np.geomspace(1e3, 1e6, 3)
OSR = np.pow(2, np.arange(2, 7)).astype(int)
N = np.arange(2, 5)
Bw = np.geomspace(1e6, 1e8, 3)
H_inf = np.array([1.5])
Nlev = np.pow(2, np.arange(1, 3)).astype(int)
Form = ["FB", "FF"]

start_time = time.time()

divides_indices = []
if rank == 0:
    configuration = list(itertools.product(Cint, Ro, OSR, N, Bw, H_inf, Nlev, Form))
    np.random.shuffle(configuration)
    number_of_configurations = len(configuration)
    print(f"Number of configurations: {number_of_configurations}")
    divides_indices = [
        list(sublist) for sublist in np.array_split(configuration, comm_size)
    ]

configuration = comm.scatter(divides_indices, root=0)

res = []
for config in configuration:
    # print(f"Rank {rank} is processing config {config}")
    cint, ro, osr, n, bw, h_inf, nlev, form = config
    cint = float(cint)
    ro = float(ro)
    osr = int(osr)
    n = int(n)
    bw = float(bw)
    h_inf = float(h_inf)
    nlev = int(nlev)
    form = str(form)
    fs = 2.0 * bw * osr

    h = ds.synthesizeNTF(n, osr, 2, h_inf)
    ABCDc, tdac = ds.realizeNTF_ct(h, form=form)
    try:
        ABCDc, _, _ = ds.scaleABCD(ABCDc, nlev=nlev)
    except Exception as e:
        logger.error(f"Error in scaling ABCDc: {e}")
    af = AnalogFrontend.ctsdm(ABCDc, tdac, 1.0 / fs, quantization_levels=nlev)
    gmc = GmC(af, ro * np.ones(af.N), cint * np.ones(af.N))

    try:
        snr, amp = gmc.simulateSNR(osr)
    except Exception as e:
        snr = np.array([0.0])

    f = gmc.fs
    while f > bw / 5.0:
        f /= 2.0
    frequency = np.array([[f]])
    gmc.analog_signal = Sinusoidal(amplitude, frequency)
    sim = gmc.simulate(1 << 13)
    avg_power = np.sum(gmc.avg_power(sim["x"]))

    res.append(
        {
            "Cint": cint,
            "Ro": ro,
            "OSR": osr,
            "N": n,
            "Bw": bw,
            "H_inf": h_inf,
            "nlev": nlev,
            "form": form,
            "snr": np.max(snr),
            "power": avg_power,
        }
    )

end_time = time.time()

configurations_per_rank = len(res)
print(
    "\n".join(
        [
            f"Rank {rank} took {end_time - start_time:0.0f} seconds"
            f"for {configurations_per_rank} simulations,"
            f"i.e., on average, {(end_time - start_time) / configurations_per_rank : 0.1f} seconds / data point"
        ]
    )
)

res = comm.gather(res, root=0)


if rank == 0:
    with open("results.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=res[0][0].keys())
        writer.writeheader()
        for rank_res in res:
            for row in rank_res:
                writer.writerow(row)
