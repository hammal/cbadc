from cbadc.verilog_ams.numbers import Number, Logic, Integer, Real


def test_number_base_class():
    assert(str(Number('any_value')) == 'any_value')
    assert(str(Number('1')) == '1')
    assert(str(Number(1)) == '1')
    assert(str(Number(1.1)) == '1.1')
