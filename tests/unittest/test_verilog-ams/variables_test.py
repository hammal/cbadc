import cbadc.verilog_ams
import cbadc.verilog_ams.variables as variables


def test_variable():
    var = variables.Variables('a_variable_name')
    assert(var.definition() == [])
    assert(str(var) == "a_variable_name")


def test_integer():
    integer = variables.Integer('a_integer_name', int(2))
    assert(str(integer) == 'a_integer_name')
    assert(integer.definition() == ['integer a_integer_name = 2;'])


def test_integer_default():
    integer = variables.Integer('another_name')
    assert(str(integer) == 'another_name')
    assert(integer.definition() == ['integer another_name = 0;'])
    assert(str(integer.initial_value) == str(
        cbadc.verilog_ams.numbers.Integer(0.0)))


def test_real_default():
    real = variables.Real('another_name')
    assert(str(real) == 'another_name')
    assert(real.definition() == ['real another_name;'])


def test_real_continuous_time():
    real = variables.Real('another_name', 1.0, continuous_time=True)
    assert(str(real) == 'another_name')
    assert(real.definition() == ['real another_name;'])
    assert(str(real.initial_value) ==
           str(cbadc.verilog_ams.numbers.Real(1.0)))


def test_real_discrete_time():
    real = variables.Real('another_name', 7.0, continuous_time=False)
    assert(str(real) == 'another_name')
    assert(real.definition() == ['real another_name = 7.0;'])


def test_real_discrete_no_initial_value_time():
    real = variables.Real('another_name',  continuous_time=False)
    assert(str(real) == 'another_name')
    assert(real.definition() == ['real another_name = 0.0;'])
