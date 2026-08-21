from battery_lab.ecm import TheveninParameters, TheveninState, coulomb_count, thevenin_step


def test_coulomb_count_discharge_reduces_soc():
    assert coulomb_count(1.0, 1.0, 360.0, 1.0) < 1.0


def test_thevenin_step():
    params = TheveninParameters(capacity_Ah=2.0, r0_ohm=0.01, r1_ohm=0.02, c1_F=1000)
    state, voltage = thevenin_step(TheveninState(1.0), 1.0, 1.0, params, 4.1)
    assert state.soc < 1.0
    assert voltage < 4.1
