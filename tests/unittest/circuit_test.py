"""Physical gmC and active-RC realizations preserve their component equations."""

import numpy as np
import pytest

from cbadc.circuit import ActiveRC, GmC
from cbadc.frontend import Frontend
from cbadc.noise import k_B


def test_gmc_realization_and_single_conversion():
    """Fixed transconductances survive added capacitance and output leakage."""
    af = Frontend(
        [[-2.0, 1.0], [0.0, -3.0]],
        [[1.0, -1.0, 0.0], [0.0, 0.0, -2.0]],
        np.eye(2),
        dt=0.1,
        D_hat=[[0.25, 0.5]],
    )
    assert not hasattr(af, "netlist")
    gmc = GmC(af, Cint=[2e-12, 3e-12], Cp=[1e-12, 0], Ro=[1e6, np.inf])
    explicit = GmC(
        af,
        Cint=[2e-12, 3e-12],
        Cp=[1e-12, 0],
        Ro=[1e6, np.inf],
        C_design=[2e-12, 3e-12],
    )
    np.testing.assert_array_equal(gmc.A, explicit.A)
    np.testing.assert_array_equal(gmc.B, explicit.B)
    np.testing.assert_array_equal(gmc.gm_A, explicit.gm_A)
    np.testing.assert_array_equal(gmc.gm_B, explicit.gm_B)
    scale = np.array([2 / 3, 1])
    np.testing.assert_allclose(
        gmc.A[0, 0], scale[:, None] * af.A[0, 0] - np.diag([1 / (3e-12 * 1e6), 0])
    )
    np.testing.assert_allclose(gmc.B[0, 0], scale[:, None] * af.B[0, 0])
    af.C[0, 0, 0, 0] = 5.0
    assert gmc.C[0, 0, 0, 0] == 1.0
    af.D_hat[0, 0, 0, 0] = 9.0
    assert gmc.D_hat[0, 0, 0, 0] == 0.25
    with pytest.raises(ValueError, match="not another circuit realization"):
        GmC(gmc)


def test_gmc_needs_continuous_time():
    """Physical C and R cannot wrap an already discrete step map."""
    af = Frontend([[1.0]], [[0.0, 0.0]], [[1.0]], dt=1.0, discrete=True)
    with pytest.raises(ValueError, match="continuous-time"):
        GmC(af)


def test_gmc_batch_member_keeps_its_transconductance():
    """Selecting ``j`` carries the same physical gm and realized ``A`` row."""
    af = Frontend(np.array([[[[-1.0]]], [[[-2.0]]]]), [[0.0, -1.0]], [[1.0]], dt=0.1)
    gmc = GmC(af, Cint=2e-12)
    member = gmc[1]
    np.testing.assert_array_equal(member.A[0], gmc.A[1])
    np.testing.assert_array_equal(member.gm_A[0], gmc.gm_A[1])
    with pytest.raises(ValueError, match="not another circuit realization"):
        GmC(member)


def test_gmc_nominal_parasitic_compensation_and_signed_mismatch():
    """Fixed gm follows design C; actual C changes both A and B by 1/Ctot."""
    A = np.array([[[-2.0, 1.0], [0.5, -3.0]], [[-4.0, 2.0], [1.0, -6.0]]])
    B = np.array(
        [[[1.0, -1.0, 0.0], [0.0, 2.0, -2.0]], [[2.0, -2.0, 0.0], [0.0, 4.0, -4.0]]]
    )
    af = Frontend(A, B, np.eye(2), dt=0.1)
    Cint_nominal = np.array([2e-12, 3e-12])
    Cp_nominal = np.array([0.4e-12, 0.6e-12])
    C_design = Cint_nominal + Cp_nominal
    Ro = np.array([1e6, 2e6])
    nominal = GmC(af, Cint=Cint_nominal, Cp=Cp_nominal, C_design=C_design, Ro=Ro)
    leakage = np.diag(1 / (Ro * C_design))
    np.testing.assert_allclose(nominal.A[0], af.A[0] - leakage)
    np.testing.assert_allclose(nominal.B[0], af.B[0])

    # Both signs of component deviation appear in each batch and phase.
    delta_int = np.array(
        [
            [[0.2e-12, -0.2e-12], [-0.1e-12, 0.1e-12]],
            [[-0.2e-12, 0.2e-12], [0.1e-12, -0.1e-12]],
        ]
    )
    delta_p = np.array(
        [
            [[-0.1e-12, 0.1e-12], [0.1e-12, -0.1e-12]],
            [[0.1e-12, -0.1e-12], [-0.1e-12, 0.1e-12]],
        ]
    )
    actual = GmC(
        af,
        Cint=Cint_nominal + delta_int,
        Cp=Cp_nominal + delta_p,
        C_design=C_design,
    )
    scale = C_design / actual.Ctot
    np.testing.assert_allclose(actual.A, scale[..., :, None] * af.A)
    np.testing.assert_allclose(actual.B, scale[..., :, None] * af.B)
    np.testing.assert_array_equal(actual.gm_A[0], actual.gm_A[1])
    np.testing.assert_array_equal(actual.gm_B[0], actual.gm_B[1])
    member = actual[1]
    for name in ("Cint", "Cp", "C_design", "Ctot", "Ro", "gm_A", "gm_B"):
        original = getattr(actual, name)
        np.testing.assert_array_equal(
            getattr(member, name)[0], original[1 % len(original)]
        )
    np.testing.assert_array_equal(member.A[0], actual.A[1])
    np.testing.assert_array_equal(member.B[0], actual.B[1])


def test_gmc_batch_components_and_netlist_use_selected_actual_values():
    """A selected Monte Carlo sample keeps its own capacitor and resistance."""
    af = Frontend([[-2.0]], [[1.0, -1.0]], [[1.0]], dt=0.1)
    gmc = GmC(
        af,
        Cint=np.array([[2e-12], [2.2e-12]]),
        Cp=np.array([[0.4e-12], [0.5e-12]]),
        Ro=np.array([[1e6], [2e6]]),
        C_design=2.4e-12,
    )
    assert (gmc.J, gmc.K) == (2, 1)
    np.testing.assert_array_equal(gmc.gm_A[0], gmc.gm_A[1])
    np.testing.assert_array_equal(gmc.gm_B[0], gmc.gm_B[1])
    selected = gmc[1]
    netlist = selected.netlist(2)
    assert f"Cx0 x0 0 {float(gmc.Ctot[1, 0, 0])!r}" in netlist
    assert "Rx0 x0 0 2000000.0" in netlist
    assert f"Gax0_0 0 x0 x0 0 {float(gmc.gm_A[1, 0, 0, 0])!r}" in netlist


def test_gmc_rejects_nonphysical_capacitances():
    """Signed mismatch is allowed only while actual components stay valid."""
    af = Frontend([[-1.0]], [[1.0, -1.0]], [[1.0]], dt=0.1)
    with pytest.raises(ValueError, match="Cint must be finite and positive"):
        GmC(af, Cint=0.0)
    with pytest.raises(ValueError, match="Cp must be finite and nonnegative"):
        GmC(af, Cp=-0.1e-12)
    with pytest.raises(ValueError, match="C_design must be finite and positive"):
        GmC(af, C_design=0.0)


def test_active_rc_two_pole_equations_and_frontend_settings():
    """Cint and the op-amp pole produce the stated 2N-state ABCD model."""
    af = Frontend(
        [[-2.0, 1.0], [0.0, -3.0]],
        [[1.0, -1.0, 0.0], [0.0, 0.0, -2.0]],
        np.eye(2),
        dt=0.1,
        C_hat=[[-1.0, 0.0]],
        D_hat=[[0.25, 0.5]],
    )
    af.state_covariance = np.diag([1e-6, 2e-6])
    af.state_min = [-1.0, -2.0]
    af.slew_rate = [3.0, 4.0]
    rc = ActiveRC(
        af,
        Cint=[2e-12, 3e-12],
        gm=[2e-3, 3e-3],
        Ro=1e6,
        Co=1e-12,
        omega_u=[5.0, 6.0],
        thermal_noise=False,
    )
    wt = (rc.gm / rc.Co)[0, 0]
    np.testing.assert_allclose(
        rc.A[0, 0, :2, :2], -np.diag(rc.omega_u / rc.DC_gain[0, 0])
    )
    np.testing.assert_allclose(rc.A[0, 0, :2, 2:], np.diag(rc.omega_u))
    np.testing.assert_allclose(
        rc.A[0, 0, 2:, :2], wt[:, None] / rc.omega_u[:, None] * af.A[0, 0]
    )
    np.testing.assert_allclose(rc.A[0, 0, 2:, 2:], -np.diag(wt))
    np.testing.assert_allclose(
        rc.B[0, 0, 2:], wt[:, None] / rc.omega_u[:, None] * af.B[0, 0]
    )
    np.testing.assert_array_equal(rc.C[0, 0, :, :2], af.C[0, 0])
    np.testing.assert_array_equal(rc.C_hat[0, 0, :, :2], af.C_hat[0, 0])
    np.testing.assert_array_equal(rc.D_hat, af.D_hat)
    np.testing.assert_array_equal(
        rc.state_covariance[0, 0, :2, :2], af.state_covariance[0, 0]
    )
    np.testing.assert_array_equal(rc.state_min, [-1.0, -2.0, -np.inf, -np.inf])
    np.testing.assert_array_equal(rc.slew_rate[:2], af.slew_rate)
    np.testing.assert_allclose(rc.R_int, 1 / (rc.omega_u * rc.C_design))
    np.testing.assert_allclose(rc.GBWP, rc.gm / (2 * np.pi * rc.Co))
    with pytest.raises(ValueError, match="not another circuit realization"):
        ActiveRC(rc, 1e-12, 1e-3, 1e6, 1e-12)
    with pytest.raises(ValueError, match="not another circuit realization"):
        GmC(rc)


def test_active_rc_low_frequency_limit_and_netlist():
    """A fast, high-gain op-amp approaches the ideal transfer function."""
    af = Frontend([[-2.0]], [[1.0, -1.0]], [[1.0]], dt=0.1)
    rc = ActiveRC(af, Cint=2e-12, gm=1e-3, Ro=1e9, Co=1e-14, omega_u=2.0)
    jw = np.array([0.1j, 1.0j])
    _, ideal = af.transfer_function(jw, input_index=0)
    _, realized = rc.transfer_function(jw, input_index=0)
    np.testing.assert_allclose(realized, ideal, rtol=1e-4)
    netlist = rc.netlist(2)
    assert "Cx0 x0 0 2e-12" in netlist
    assert "Cop0 x1 0 1e-14" in netlist
    assert "Rleak0 x0 0" in netlist
    assert "Rop0 x1 0 1000.0" in netlist
    assert "Gxp0 0 x0 x1 0" in netlist
    assert "Gax0_0 0 x1 x0 0" in netlist
    assert "wrdata tran.dat v(x0) v(x1) v(v0)" in netlist


def test_active_rc_infinite_ro_removes_only_integrator_leak():
    """A one-stage integrator keeps its finite op-amp pole when ``Ro=inf``."""
    af = Frontend([[0.0]], [[1.0, -1.0]], [[1.0]], dt=0.1)
    gm, Co, omega_u = 1e-3, 1e-12, 2.0
    common = dict(Cint=2e-12, gm=gm, Co=Co, omega_u=omega_u)
    finite = ActiveRC(af, Ro=1e6, **common)
    infinite = ActiveRC(af, Ro=np.inf, **common)

    np.testing.assert_allclose(finite.A[0, 0, 0, 0], -omega_u / (gm * 1e6))
    np.testing.assert_allclose(infinite.A[0, 0, 0, 0], 0.0, atol=0, rtol=0)
    np.testing.assert_array_equal(infinite.A[0, 0, 1], finite.A[0, 0, 1])
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvals(infinite.A[0, 0])), [-gm / Co, 0.0]
    )
    assert "Rleak0 x0 0" in finite.netlist(2)
    assert "Rleak0" not in infinite.netlist(2)


def test_active_rc_fixed_resistor_with_capacitance_mismatch():
    """Both signs of mismatch alter the integrator rate, not its resistor."""
    af = Frontend([[-2.0]], [[1.0, -1.0]], [[1.0]], dt=0.1)
    C_design = 2.5e-12  # nominal Cint + nominal Cp
    common = dict(gm=1e-3, Ro=1e6, Co=1e-12, omega_u=2.0)
    nominal = ActiveRC(af, Cint=2e-12, Cp=0.5e-12, C_design=C_design, **common)
    baseline = ActiveRC(af, Cint=C_design, **common)
    np.testing.assert_array_equal(nominal.A, baseline.A)
    np.testing.assert_array_equal(nominal.B, baseline.B)

    actual = ActiveRC(
        af,
        Cint=np.array([[1.8e-12], [2e-12], [2.2e-12]]),
        Cp=np.array([[0.4e-12], [0.5e-12], [0.6e-12]]),
        C_design=C_design,
        **common,
    )
    assert (actual.J, actual.K) == (3, 1)
    np.testing.assert_allclose(actual.R_int, np.broadcast_to(nominal.R_int, (3, 1, 1)))
    rate = 1 / (actual.R_int * actual.Ctot)
    np.testing.assert_allclose(actual.A[:, :, 0, 1], rate[:, :, 0])
    np.testing.assert_allclose(
        actual.A[:, :, 0, 0], -rate[:, :, 0] / actual.DC_gain[:, :, 0]
    )
    assert rate[0, 0, 0] > rate[1, 0, 0] > rate[2, 0, 0]
    np.testing.assert_array_equal(
        actual.A[:, :, 1:, :1], np.broadcast_to(nominal.A[:, :, 1:, :1], (3, 1, 1, 1))
    )
    np.testing.assert_array_equal(actual.B, np.broadcast_to(nominal.B, actual.B.shape))
    selected = actual[2]
    for name in ("Cint", "Cp", "C_design", "Ctot", "R_int", "DC_gain"):
        original = getattr(actual, name)
        np.testing.assert_array_equal(
            getattr(selected, name)[0], original[2 % len(original)]
        )
    netlist = selected.netlist(2)
    assert f"Cx0 x0 0 {float(actual.Ctot[2, 0, 0])!r}" in netlist
    assert f"Gxp0 0 x0 x1 0 {float(1 / actual.R_int[2, 0, 0])!r}" in netlist


def test_active_rc_batched_clock_phases_keep_matching_capacitances():
    """The ``(J, K)`` component map survives selection of a batch member."""
    af = Frontend([[-2.0]], [[1.0, -1.0]], [[1.0]], dt=0.1)
    Cint = np.array([[[1.8e-12], [2.2e-12]], [[2.1e-12], [1.9e-12]]])
    rc = ActiveRC(
        af,
        Cint=Cint,
        Cp=0.2e-12,
        C_design=2e-12,
        gm=1e-3,
        Ro=1e6,
        Co=1e-12,
        omega_u=2.0,
    )
    assert (rc.J, rc.K) == (2, 2)
    np.testing.assert_allclose(rc.A[:, :, 0, 1], 2e-12 / rc.Ctot[:, :, 0] * 2)
    member = rc[1]
    np.testing.assert_array_equal(member.Cint[0], rc.Cint[1])
    np.testing.assert_array_equal(member.A[0], rc.A[1])


def test_active_rc_rejects_discrete_and_invalid_components():
    """Only a continuous ideal frontend and finite positive RC values apply."""
    dt = Frontend([[1.0]], [[0.0, -1.0]], [[1.0]], dt=1.0, discrete=True)
    with pytest.raises(ValueError, match="continuous-time"):
        ActiveRC(dt, 1e-12, 1e-3, 1e6, 1e-12)
    ct = Frontend([[-1.0]], [[1.0, -1.0]], [[1.0]], dt=0.1)
    with pytest.raises(ValueError, match="Co must be finite and positive"):
        ActiveRC(ct, 1e-12, 1e-3, 1e6, 0)
    with pytest.raises(ValueError, match="Cp must be finite and nonnegative"):
        ActiveRC(ct, 1e-12, 1e-3, 1e6, 1e-12, Cp=-1e-13)
    with pytest.raises(ValueError, match="Ro must be positive or infinite"):
        ActiveRC(ct, 1e-12, 1e-3, np.nan, 1e-12)


def test_active_rc_preserves_batch_and_phase_matrices():
    """Each ``(j, k)`` augmented row uses its own ideal ``A, B`` values."""
    A = np.array([[[[-1.0]], [[-2.0]]], [[[-3.0]], [[-4.0]]]])
    B = np.array([[[[1.0, -1.0]], [[2.0, -2.0]]], [[[3.0, -3.0]], [[4.0, -4.0]]]])
    af = Frontend(A, B, [[1.0]], dt=0.1)
    rc = ActiveRC(af, Cint=1e-12, gm=1e-3, Ro=1e6, Co=1e-12)
    assert (rc.J, rc.K, rc.N) == (2, 2, 2)
    np.testing.assert_allclose(rc.omega_u, [4.0])
    np.testing.assert_allclose(rc.A[:, :, 1, 0], 250_000_000 * A[:, :, 0, 0])
    np.testing.assert_allclose(rc.B[:, :, 1], 250_000_000 * B[:, :, 0])
    np.testing.assert_array_equal(rc[1].A[0], rc.A[1])


def test_gmc_thermal_noise_uses_fixed_gm_and_actual_capacitance():
    """Each (j, k) noise intensity follows its own Ctot and finite Ro."""
    af = Frontend([[-2.0]], [[1.0, -1.0]], [[1.0]], dt=0.01)
    af.state_covariance = [[0.25]]
    cint = np.array([[[2e-12], [3e-12]], [[4e-12], [5e-12]]])
    gmc = GmC(
        af,
        Cint=cint,
        C_design=2e-12,
        Ro=1e6,
        thermal_noise=True,
        temperature=300.0,
        gamma=1.5,
    )
    gm_sum = np.abs(gmc.gm_A).sum(-1) + np.abs(gmc.gm_B).sum(-1)
    expected = 0.25 + 2 * k_B * 300 * (1.5 * gm_sum + 1e-6) / cint**2
    np.testing.assert_allclose(gmc.state_covariance[..., 0, 0], expected[..., 0])
    np.testing.assert_array_equal(gmc[1].state_covariance[0], gmc.state_covariance[1])
    np.testing.assert_array_equal(
        GmC(af).state_covariance, GmC(af, thermal_noise=True).state_covariance
    )
    np.testing.assert_array_equal(
        GmC(af, thermal_noise=False).state_covariance, af.state_covariance
    )
    with pytest.raises(ValueError, match="thermal_noise or v_n"):
        GmC(af, thermal_noise=True, v_n=1e-9)


def test_active_rc_thermal_and_opamp_noise_enter_different_states():
    """R_int and input current drive x; input voltage drives the pole p."""
    af = Frontend([[-2.0]], [[1.0, -1.0]], [[1.0]], dt=0.01)
    rc = ActiveRC(
        af,
        Cint=np.array([[2e-12], [3e-12]]),
        C_design=2e-12,
        gm=1e-3,
        Ro=np.inf,
        Co=1e-12,
        omega_u=2.0,
        thermal_noise=True,
        opamp_voltage_noise=np.array([[2e-9], [3e-9]]),
        opamp_current_noise=4e-12,
    )
    omega_t = rc.gm / rc.Co
    expected_x = (4 * k_B * 300 / rc.R_int + (4e-12) ** 2) / (2 * rc.Ctot**2)
    expected_p = (omega_t * rc.opamp_voltage_noise) ** 2 / 2
    np.testing.assert_allclose(rc.state_covariance[..., 0, 0], expected_x[..., 0])
    np.testing.assert_allclose(rc.state_covariance[..., 1, 1], expected_p[..., 0])
    np.testing.assert_array_equal(rc.state_covariance[..., 0, 1], 0)
    np.testing.assert_array_equal(rc[1].state_covariance[0], rc.state_covariance[1])
    assert "Bwn0" in rc[0].netlist(2)
    assert "Bwn1" in rc[0].netlist(2)
    np.testing.assert_array_equal(
        ActiveRC(af, 2e-12, 1e-3, 1e6, 1e-12).state_covariance,
        ActiveRC(af, 2e-12, 1e-3, 1e6, 1e-12, thermal_noise=True).state_covariance,
    )
    assert (
        ActiveRC(af, 2e-12, 1e-3, 1e6, 1e-12, thermal_noise=False).state_covariance
        is None
    )
    with pytest.raises(ValueError, match="circuit noise parameters or v_n"):
        ActiveRC(af, 2e-12, 1e-3, 1e6, 1e-12, v_n=1e-9, thermal_noise=True)


@pytest.mark.parametrize("realization", ["gmc", "active_rc"])
def test_circuit_comparator_noise_is_input_referred_per_decision(realization):
    """Each comparator gets independent RMS noise added to inherited covariance."""
    af = Frontend([[-2.0]], [[1.0, -1.0, -1.0]], [[1.0], [1.0]], dt=0.01)
    af.output_covariance = np.array([[1e-6, 2e-7], [2e-7, 3e-6]])
    kwargs = {"comparator_input_noise_rms": [1e-3, 2e-3]}
    circuit = (
        GmC(af, **kwargs)
        if realization == "gmc"
        else ActiveRC(af, 2e-12, 1e-3, 1e6, 1e-12, **kwargs)
    )
    np.testing.assert_allclose(
        circuit.output_covariance, af.output_covariance + np.diag([1e-6, 4e-6])
    )
    np.testing.assert_array_equal(
        circuit[0].output_covariance, circuit.output_covariance
    )
    assert "Veta0" in circuit.netlist(2)
    with pytest.raises(ValueError, match="comparator_input_noise_rms"):
        if realization == "gmc":
            GmC(af, comparator_input_noise_rms=[-1, 0])
        else:
            ActiveRC(af, 2e-12, 1e-3, 1e6, 1e-12, comparator_input_noise_rms=[-1, 0])
