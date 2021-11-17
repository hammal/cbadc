import cbadc.verilog_ams.expressions
import cbadc.verilog_ams.variables
import cbadc.verilog_ams.nets
import unittest
import pytest


class BaseExpressionTestClass(unittest.TestCase):

    def test_base_expression(self):
        name = "asldkjaslkd"
        expr = cbadc.verilog_ams.expressions.BaseExpression(name)
        assert(str(expr) == name)
        assert(expr.continuous_kernel == False)
        assert(bool(expr) == False)
        assert(len(expr) == 1)

    def test_base_expression_ck(self):
        name = "aljksadl"
        expr = cbadc.verilog_ams.expressions.BaseExpression(
            continuous_kernel=True)
        assert(len(expr) == 1)
        assert(str(expr) == '')
        assert(expr.continuous_kernel == True)
        assert(bool(expr) == False)

    def test_argument_expression_checker(self):
        names = 'a b c d'.split()
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        aec = cbadc.verilog_ams.expressions._ArgumentExpressionChecker(*expr)
        assert(len(aec) == 4)
        assert(bool(aec) == True)
        assert(aec.continuous_kernel == False)
        assert(str(aec) == 'a b c d')

    def test_argument_expression_checker_raise_exception(self):
        names = 'a b c d'.split()
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        expr.append([str("Hello World")])
        with self.assertRaises(BaseException):
            cbadc.verilog_ams.expressions._ArgumentExpressionChecker(*expr)
        expr = expr[:4]
        expr[1].continuous_kernel = True
        with self.assertRaises(BaseException):
            cbadc.verilog_ams.expressions._ArgumentExpressionChecker(*expr)

    def test_parenthesis(self):
        name = "aljksadl"
        expr = cbadc.verilog_ams.expressions.BaseExpression(name)
        parethesis = cbadc.verilog_ams.expressions.Parenthesis(expr)
        assert(len(parethesis) == 1)
        assert(str(parethesis) == f"({name})")
        assert(parethesis.continuous_kernel == False)
        assert(bool(parethesis) == True)

    def test_add(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        add = cbadc.verilog_ams.expressions.Add(expr[0], expr[1])
        assert(len(add) == 3)
        assert(str(add) == f"{names[0]} + {names[1]}")

    def test_final_expressiontract(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.Subtract(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} - {names[1]}")

    def test_multiply(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.Multiply(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} * {names[1]}")

    def test_divide(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.Divide(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} / {names[1]}")

    def test_power(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.Power(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} ** ({names[1]})")

    def test_modulus(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.Modulus(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} % {names[1]}")

    def test_BitwiseAnd(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.BitwiseAnd(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} & {names[1]}")

    def test_BitwiseOr(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.BitwiseOr(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} | {names[1]}")

    def test_BitwiseXor(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.BitwiseXor(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} ^ {names[1]}")

    def test_BitwiseNot(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.BitwiseNot(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} ~ {names[1]}")

    def test_AndOperator(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.AndOperator(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} && {names[1]}")

    def test_OrOperator(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.OrOperator(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} || {names[1]}")

    def test_EqualTo(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.EqualTo(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} == {names[1]}")

    def test_NotEqualTo(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.NotEqualTo(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} != {names[1]}")

    def test_IdenticalTo(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.IdenticalTo(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} === {names[1]}")

    def test_LessThan(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.LessThan(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} < {names[1]}")

    def test_GreaterThan(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.GreaterThan(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} > {names[1]}")

    def test_LessThanOrEqualTo(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.LessThanOrEqualTo(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} <= {names[1]}")

    def test_GreaterThanOrEqualTo(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.GreaterThanOrEqualTo(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} >= {names[1]}")

    def test_LeftShift(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.LeftShift(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} << {names[1]}")

    def test_RightShift(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.RightShift(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} >> {names[1]}")

    def test_ArithmeticRightShift(self):
        names = ["a", "b"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.ArithmeticRightShift(
            expr[0], expr[1])
        assert(len(final_expression) == 3)
        assert(str(final_expression) == f"{names[0]} >>> {names[1]}")

    def test_BeginEnd(self):
        names = ["a", "b", "c", "d"]
        expr = [cbadc.verilog_ams.expressions.BaseExpression(
            name) for name in names]
        final_expression = cbadc.verilog_ams.expressions.BeginEnd(
            *expr)
        assert(len(final_expression) == 4)
        assert(str(final_expression) == "begin\n\ta;\n\tb;\n\tc;\n\td;\nend")


@pytest.fixture
def generate_expressions():
    names = ["first", "second", "third", "forth", "fifth"]
    base_expressions = [
        cbadc.verilog_ams.expressions.BaseExpression(name) for name in names]
    real_variables = [cbadc.verilog_ams.variables.Real('real_'+name,
                                                       2.12 * number) for number, name in enumerate(names)]
    real_variables_discrete_time = [cbadc.verilog_ams.variables.Real('real_'+name,
                                                                     2.12 * number, continuous_time=False) for number, name in enumerate(names)]
    electric_wires = [cbadc.verilog_ams.nets.Electrical(
        'electrical_' + name) for name in names]
    return (base_expressions, real_variables, real_variables_discrete_time, electric_wires)


def test_expression_initialization(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    assert(len(expr) == 1)
    assert(expr.continuous_kernel == False)
    assert(str(expr) == '')
    assert(expr.empty() == True)


def test_expression__add__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr + base[0]
    assert(str(result) == f' + {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result += w
    for r in real:
        more_complicated_result += r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " + ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result += r
    for b in base:
        even_more_complicated_result += b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " + ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__sub__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr - base[0]
    assert(str(result) == f' - {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result -= w
    for r in real:
        more_complicated_result -= r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " - ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result -= r
    for b in base:
        even_more_complicated_result -= b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " - ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__mul__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr * base[0]
    assert(str(result) == f' * {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result *= w
    for r in real:
        more_complicated_result *= r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " * ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result *= r
    for b in base:
        even_more_complicated_result *= b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " * ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__truediv__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr / base[0]
    assert(str(result) == f' / {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result /= w
    for r in real:
        more_complicated_result /= r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " / ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result /= r
    for b in base:
        even_more_complicated_result /= b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " / ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__pow__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr ** base[0]
    assert(str(result) == f' ** ({base[0]})')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result = more_complicated_result ** w
    for r in real:
        more_complicated_result = more_complicated_result ** r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    # assert(str(more_complicated_result) ==
    #        " ** ( ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result = even_more_complicated_result ** r
    for b in base:
        even_more_complicated_result = even_more_complicated_result ** b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    # assert(str(even_more_complicated_result) ==
    #        " * ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__mod__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr % base[0]
    assert(str(result) == f' % {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result %= w
    for r in real:
        more_complicated_result %= r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " % ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result %= r
    for b in base:
        even_more_complicated_result %= b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " % ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__lshift__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr << base[0]
    assert(str(result) == f' << {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result = more_complicated_result << w
    for r in real:
        more_complicated_result = more_complicated_result << r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " << ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result = even_more_complicated_result << r
    for b in base:
        even_more_complicated_result = even_more_complicated_result << b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " << ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__rshift__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr >> base[0]
    assert(str(result) == f' >> {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result = more_complicated_result >> w
    for r in real:
        more_complicated_result = more_complicated_result >> r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " >> ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result = even_more_complicated_result >> r
    for b in base:
        even_more_complicated_result = even_more_complicated_result >> b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " >> ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__and__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr & base[0]
    assert(str(result) == f' && {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result = more_complicated_result & w
    for r in real:
        more_complicated_result = more_complicated_result & r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " && ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result = even_more_complicated_result & r
    for b in base:
        even_more_complicated_result = even_more_complicated_result & b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " && ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__or__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr | base[0]
    assert(str(result) == f' || {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result = more_complicated_result | w
    for r in real:
        more_complicated_result = more_complicated_result | r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " || ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result = even_more_complicated_result | r
    for b in base:
        even_more_complicated_result = even_more_complicated_result | b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " || ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__xor__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr ^ base[0]
    assert(str(result) == f' ^ {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:]:
        more_complicated_result = more_complicated_result ^ w
    for r in real:
        more_complicated_result = more_complicated_result ^ r

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 10 + 9)
    assert(str(more_complicated_result) ==
           " ^ ".join([str(w) for w in wire + real]))
    assert(more_complicated_result.continuous_kernel == True)

    even_more_complicated_result = real_discrete[0]
    for r in real_discrete[1:]:
        even_more_complicated_result = even_more_complicated_result ^ r
    for b in base:
        even_more_complicated_result = even_more_complicated_result ^ b

    assert(even_more_complicated_result.empty() == False)
    assert(len(even_more_complicated_result) == 10 + 9)
    assert(str(even_more_complicated_result) ==
           " ^ ".join([str(w) for w in real_discrete + base]))
    assert(even_more_complicated_result.continuous_kernel == False)


def test_expression__lt__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr < base[0]
    assert(str(result) == f' < {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:2]:
        more_complicated_result = more_complicated_result < w

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 2 + 1)
    assert(str(more_complicated_result) ==
           " < ".join([str(w) for w in wire[:2]]))
    assert(more_complicated_result.continuous_kernel == True)


def test_expression__le__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr <= base[0]
    assert(str(result) == f' <= {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:2]:
        more_complicated_result = more_complicated_result <= w

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 2 + 1)
    assert(str(more_complicated_result) ==
           " <= ".join([str(w) for w in wire[:2]]))
    assert(more_complicated_result.continuous_kernel == True)


def test_expression__eq__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr == base[0]
    assert(str(result) == f' == {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:2]:
        more_complicated_result = more_complicated_result == w
    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 2 + 1)
    assert(str(more_complicated_result) ==
           " == ".join([str(w) for w in wire[:2]]))
    assert(more_complicated_result.continuous_kernel == True)


def test_expression__ne__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr != base[0]
    assert(str(result) == f' != {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:2]:
        more_complicated_result = more_complicated_result != w

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 2 + 1)
    assert(str(more_complicated_result) ==
           " != ".join([str(w) for w in wire[:2]]))
    assert(more_complicated_result.continuous_kernel == True)


def test_expression__gt__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr > base[0]
    assert(str(result) == f' > {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:2]:
        more_complicated_result = more_complicated_result > w

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 2 + 1)
    assert(str(more_complicated_result) ==
           " > ".join([str(w) for w in wire[:2]]))
    assert(more_complicated_result.continuous_kernel == True)


def test_expression__ge__(generate_expressions):
    expr = cbadc.verilog_ams.expressions.Expression()
    base, real, real_discrete, wire = generate_expressions
    result = expr >= base[0]
    assert(str(result) == f' >= {base[0]}')
    assert(len(result) == 3)
    # .empty() should return true since
    # expr is an empty expression.
    assert(result.empty() == True)

    more_complicated_result = wire[0]
    for w in wire[1:2]:
        more_complicated_result = more_complicated_result >= w

    # print(more_complicated_result)
    assert(more_complicated_result.empty() == False)
    assert(len(more_complicated_result) == 2 + 1)
    assert(str(more_complicated_result) ==
           " >= ".join([str(w) for w in wire[:2]]))
    assert(more_complicated_result.continuous_kernel == True)


def test_sin(generate_expressions):
    base, real, real_discrete, wire = generate_expressions
    sin = cbadc.verilog_ams.expressions.Sin(real[0])
    assert(sin.empty() == False)
    assert(len(sin) == 1)
    assert(str(sin) == f'sin({real[0]})')
