"""Base expressions for building the verilog am/s syntax tree.

https://verilogams.com/refman/basics/expressions.html
"""
from typing import List
from cbadc.verilog_ams.keywords import raise_exception_for_keyword


class BaseExpression():
    """Basic expression from which all valid expression build.

    Parameters
    ----------
    name: `str`, `optional`
        the name of the expressions. This is also
        the name that will be rendered when calling __str__().
        Defaults to empty string.
    continuous_kernel: `bool`, `optional`
        a bool describing if the given expression is to be associated
        with the verilog-ams continuous-time or discrete-time kernel.
        Defaults to False.

    Attributes
    ----------
    continuous_kernel: `bool`
        a bool describing if the given expression is to be associated
        with the verilog-ams continuous-time or discrete-time kernel.
    leaves: `list[:py:class:`cbadc.verilog_ams.expressions.BaseExpression`]
        a list of all sub expression associated with this expression.
    """

    def __init__(self, name: str = '', continuous_kernel=False) -> None:
        self.continuous_kernel = continuous_kernel
        self._name = str(name)
        # raise_exception_for_keyword(self._name)
        self.leaves: List[BaseExpression] = []

    def __str__(self) -> str:
        if self:
            return " ".join([str(leaf) for leaf in self.leaves])
        return self._name

    def __bool__(self):
        return len(self.leaves) > 0

    def __len__(self):
        if not self:
            return 1
        sum = 0
        for l in self.leaves:
            sum += len(l)
        return sum

    def __hash__(self):
        return hash((self._name, self.leaves, self.continuous_kernel))

    def empty(self):
        """Determines if this, or any sub-expressions,
        are empty.

        Returns
        -------
        : `bool`
            True if this or any sub-expressions are empty (undefined).
        """
        if self:
            for leaf in self.leaves:
                if leaf.empty():
                    return True
            return False
        if str(self) == '':
            return True
        return False


class _ArgumentExpressionChecker(BaseExpression):

    def __init__(self, *args: BaseExpression) -> None:
        super().__init__()
        self.leaves = [*args]
        if self:
            self.continuous_kernel = self.leaves[0].continuous_kernel
        self._check_args()

    def _check_args(self):
        # Check that all correctly derive
        for arg in self.leaves:
            if not isinstance(arg, BaseExpression):
                raise BaseException(
                    f"{arg} must derive from BaseExpression and not {type(arg)}")
            if self.continuous_kernel != arg.continuous_kernel:
                raise BaseException(
                    "All arguments have same continuous_time kernel")


class Parenthesis(_ArgumentExpressionChecker):
    """Generate a parenthesis expression encapsulating any
    arguments as sub-expressions, i.e.,

    For constructing a verilog am/s block like::

        ( expression )

    Parameters
    ----------
    Parameters
    ----------
    expression: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        the expression to be embedded in parenthesis.
    """

    def __init__(self, a: BaseExpression):
        super().__init__(a)

    def __str__(self) -> str:
        return f"({super().__str__()})"


class Add(_ArgumentExpressionChecker):
    """Add two expressions.

    Returns the result of adding two expressions as

    For constructing a verilog am/s block like::

        a + b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("+", continuous_kernel=self.continuous_kernel),
            self.leaves[1]
        ]


class Subtract(_ArgumentExpressionChecker):
    """Subtract one expression from another.

    For constructing a verilog am/s block like::

        a - b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("-", continuous_kernel=self.continuous_kernel),
            self.leaves[1]
        ]


class Multiply(_ArgumentExpressionChecker):
    """Multiply two expressions.

    For constructing a verilog am/s block like::

        a * b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("*", continuous_kernel=self.continuous_kernel),
            self.leaves[1]
        ]


class Divide(_ArgumentExpressionChecker):
    """Divide one expression by another.

    For constructing a verilog am/s block like::

        a / b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("/", continuous_kernel=self.continuous_kernel),
            self.leaves[1]
        ]


class Power(_ArgumentExpressionChecker):
    """Take the power one expression to another.

    For constructing a verilog am/s block like::

        a ** b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("**", continuous_kernel=self.continuous_kernel),
            Parenthesis(self.leaves[1])
        ]


class Modulus(_ArgumentExpressionChecker):
    """The modulus operator.

    For constructing a verilog am/s block like::

        a % b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("%", continuous_kernel=self.continuous_kernel),
            self.leaves[1]
        ]


class BitwiseAnd(_ArgumentExpressionChecker):
    """The bitwise and operator.

    For constructing a verilog am/s block like::

        a & b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("&", continuous_kernel=False),
            self.leaves[1]
        ]


class BitwiseOr(_ArgumentExpressionChecker):
    """The bitwise or operator.

    For constructing a verilog am/s block like::
        a | b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("|", continuous_kernel=False),
            self.leaves[1]
        ]


class BitwiseXor(_ArgumentExpressionChecker):
    """The bitwise xor operator.

    For constructing a verilog am/s block like::

        a ^ b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("^", continuous_kernel=False),
            self.leaves[1]
        ]


class BitwiseNot(_ArgumentExpressionChecker):
    """The bitwise not operator.

    For constructing a verilog am/s block like::

        a ~ b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("~", continuous_kernel=False),
            self.leaves[1]
        ]


class AndOperator(_ArgumentExpressionChecker):
    """The and operator.

    For constructing a verilog am/s block like::
        a && b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("&&", continuous_kernel=False),
            self.leaves[1]
        ]


class OrOperator(_ArgumentExpressionChecker):
    """The or operator.

    For constructing a verilog am/s block like::
        a || b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("||", continuous_kernel=False),
            self.leaves[1]
        ]


class EqualTo(_ArgumentExpressionChecker):
    """The equality to operator.

    For constructing a verilog am/s block like::
        a == b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("==", continuous_kernel=False),
            self.leaves[1]
        ]


class NotEqualTo(_ArgumentExpressionChecker):
    """The not equal to operator.

    For constructing a verilog am/s block like::
        a != b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("!=", continuous_kernel=False),
            self.leaves[1]
        ]


class IdenticalTo(_ArgumentExpressionChecker):
    """The identical to operator.

    For constructing a verilog am/s block like::
        a === b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("===", continuous_kernel=False),
            self.leaves[1]
        ]


class NotIdenticalTo(_ArgumentExpressionChecker):
    """The not identical to operator.

    For constructing a verilog am/s block like::

        a !== b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("!==", continuous_kernel=False),
            self.leaves[1]
        ]


class LessThan(_ArgumentExpressionChecker):
    """The less than operator.

    For constructing a verilog am/s block like::

        a < b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("<", continuous_kernel=False),
            self.leaves[1]
        ]


class GreaterThan(_ArgumentExpressionChecker):
    """The greater than operator.

    a > b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression(">", continuous_kernel=False),
            self.leaves[1]
        ]


class LessThanOrEqualTo(_ArgumentExpressionChecker):
    """The less than or equal to operator.

    For constructing a verilog am/s block like::
        a <= b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("<=", continuous_kernel=False),
            self.leaves[1]
        ]


class GreaterThanOrEqualTo(_ArgumentExpressionChecker):
    """The greather than or equal to operator.

    For constructing a verilog am/s block like::

        a >= b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression(">=", continuous_kernel=False),
            self.leaves[1]
        ]


class LeftShift(_ArgumentExpressionChecker):
    """The left shift operator.

    For constructing a verilog am/s block like::
        a << b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression("<<", continuous_kernel=False),
            self.leaves[1]
        ]


class RightShift(_ArgumentExpressionChecker):
    """The right shift operator.

    For constructing a verilog am/s block like::

        a >> b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression(">>", continuous_kernel=False),
            self.leaves[1]
        ]


class ArithmeticRightShift(_ArgumentExpressionChecker):
    """The arithmetic right shift operator.

    in contrast to :py:class:`cbadc.verilog_ams.expressions.RightShift`
    this performs right shift with preserved sign bit.

    For constructing a verilog am/s block like::

        a >>> b

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        second argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self.leaves = [
            self.leaves[0],
            BaseExpression(">>>", continuous_kernel=False),
            self.leaves[1]
        ]


class Expression(_ArgumentExpressionChecker):
    """The Expression class.

    This is a convenience class that maps
    the standard arithemtic object operations
    of python to the general operators inside
    this module.

    Examples
    --------
    >>> from cbadc.verilog_ams.expressions import BaseExpression, Expression
    >>> a = Expression(BaseExpression('a_node'))
    >>> b = Expression(BaseExpression('b_node'))
    >>> c = a + b
    >>> print(c)
    a_node + b_node
    """

    def __init__(self, *a: BaseExpression):
        super().__init__(*a)

    def __add__(self, other):
        return Expression(Add(self, other))

    def __sub__(self, other):
        return Expression(Subtract(self, other))

    def __mul__(self, other):
        return Expression(Multiply(self, other))

    def __truediv__(self, other):
        return Expression(Divide(self, other))

    def __pow__(self, other):
        return Expression(Power(self, other))

    def __mod__(self, other):
        return Expression(Modulus(self, other))

    def __lshift__(self, other):
        return Expression(LeftShift(self, other))

    def __rshift__(self, other):
        return Expression(RightShift(self, other))

    def __and__(self, other):
        return Expression(AndOperator(self, other))

    def __or__(self, other):
        return Expression(OrOperator(self, other))

    def __xor__(self, other):
        return Expression(BitwiseXor(self, other))

    def __lt__(self, other):
        return Expression(LessThan(self, other))

    def __le__(self, other):
        return Expression(LessThanOrEqualTo(self, other))

    def __eq__(self, other):
        return Expression(EqualTo(self, other))

    def __ne__(self, other):
        return Expression(NotEqualTo(self, other))

    def __gt__(self, other):
        return Expression(GreaterThan(self, other))

    def __ge__(self, other):
        return Expression(GreaterThanOrEqualTo(self, other))


class AbsTime(Expression):
    """The absolute time of the simulation

    For constructing a verilog am/s block like::

        $abstime

    """

    def __init__(self) -> None:
        super().__init__(BaseExpression("$abstime"))


class RealTime(Expression):
    """The real time of the simulation in verilog time units

    For constructing a verilog am/s block like::

        $realtime

    """

    def __init__(self) -> None:
        super().__init__(BaseExpression("$realtime"))


class Temperature(Expression):
    """The temperature

    For constructing a verilog am/s block like::

        $temperature

    """

    def __init__(self) -> None:
        super().__init__(BaseExpression("$temperature"))


class ThermalVoltage(Expression):
    """The thermal voltage vt

    For constructing a verilog am/s block like::

        $vt

    """

    def __init__(self) -> None:
        super().__init__(BaseExpression("$vt"))


class BeginEnd(_ArgumentExpressionChecker):
    """A begin end statement.

    Put a list of expressions inside a being end block
    as::

        begin
            expression_1;
            .
            .
            .
            expression_N;
        end

    Parameters
    ----------
    a: list[BaseExpressions]
        a list of expressions to be put inside
        the begin end block.
    """

    def __init__(self, *args: BaseExpression) -> None:
        super().__init__(*args)

    def __str__(self) -> str:
        return "begin\n\t" + ";\n\t".join([str(leaf) for leaf in self.leaves]) + ";\nend"


class _Function(Expression):
    def __init__(self, *a: BaseExpression):
        super().__init__(*a)
        self._function_name = ''

    def __str__(self) -> str:
        return f"{self._function_name}({super().__str__()})"


class Ln(_Function):
    """The natural logarithm function

    For constructing a verilog am/s block like::

        ln(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "ln"


class Log(_Function):
    """The 10 base logarithm function

    For constructing a verilog am/s block like::

        log(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "log"


class Exp(_Function):
    """The exponential function

    For constructing a verilog am/s block like::

        exp(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "exp"


class Sqrt(_Function):
    """The square root function

    For constructing a verilog am/s block like::

        sqrt(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "sqrt"


class Min(_Function):
    """The minimum function

    For constructing a verilog am/s block like::

        min(a,b)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self._function_name = "min"


class Max(_Function):
    """The maximum function

    For constructing a verilog am/s block like::

        max(a,b)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    b: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self._function_name = "max"


class Abs(_Function):
    """The absolute value function

    For constructing a verilog am/s block like::

        abs(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "abs"


class Floor(_Function):
    """The floor function

    For constructing a verilog am/s block like::

        floor(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "floor"


class Ceil(_Function):
    """The ceil function

    For constructing a verilog am/s block like::

        ceil(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "ceil"


class Sin(_Function):
    """The sin function

    For constructing a verilog am/s block like::

        sin(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "sin"


class Cos(_Function):
    """The cosine function

    For constructing a verilog am/s block like::

        cos(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "cos"


class Tan(_Function):
    """The tangence function

    For constructing a verilog am/s block like::

        tan(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "tan"


class ASin(_Function):
    """The arc sin function

    For constructing a verilog am/s block like::

        asin(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "asin"


class ACos(_Function):
    """The arc cos function

    For constructing a verilog am/s block like::

        acos(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "acos"


class ATan(_Function):
    """The arc tan function

    For constructing a verilog am/s block like::

        atan(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "atan"


class ATan2(_Function):
    """The arc tan2 function

    Angle from origin to the point (a,b)

    For constructing a verilog am/s block like::

        atan2(b, a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(b, a)
        self._function_name = "atan2"


class Hypot(_Function):
    """The distance from origin to the point (a,b)

    in other words::

        hypot(a,b) = sqrt(a**2 + b**2)


    For constructing a verilog am/s block like::

        hypot(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression, b: BaseExpression) -> None:
        super().__init__(a, b)
        self._function_name = "hypot"


class SinH(_Function):
    """The hyperbolic sine function

    For constructing a verilog am/s block like::

        sinh(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "sinh"


class CosH(_Function):
    """The hyperbolic cosine function

    For constructing a verilog am/s block like::

        cosh(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "cosh"


class TanH(_Function):
    """The hyperbolic tangence function

    For constructing a verilog am/s block like::

        tanh(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "tanh"


class ASinH(_Function):
    """The hyperbolic arc sin function

    For constructing a verilog am/s block like::

        asinh(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "asinh"


class ACosH(_Function):
    """The hyperbolic arc cos function

    For constructing a verilog am/s block like::

        acosh(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "acosh"


class ATanH(_Function):
    """The hyperbolic arc tan function

    For constructing a verilog am/s block like::

        atanh(a)

    Parameters
    ----------
    a: :py:class:`cbadc.verilog_ams.expressions.BaseExpression`
        first argument
    """

    def __init__(self, a: BaseExpression) -> None:
        super().__init__(a)
        self._function_name = "atanh"


# TODO All Function Expressions

# TODO all assignments

# TODO all documentation

# TODO all tests
