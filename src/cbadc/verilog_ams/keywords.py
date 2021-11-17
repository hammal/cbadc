"""keyword and syntax checking.
"""

__keywords = set(
    (
        "above",
        "disable",
        "idt",
        "notif1",
        "supply0",
        "abs",
        "discipline",
        "idtmod",
        "or",
        "supply1",
        "absdelay",
        "driver_update",
        "if",
        "output",
        "table",
        "ac_stim",
        "edge",
        "ifnone",
        "parameter",
        "tan",
        "acos",
        "else",
        "inf",
        "pmos",
        "tanh",
        "acosh",
        "end",
        "initial",
        "posedge",
        "task",
        "always",
        "endcase",
        "initial_step",
        "potential",
        "time",
        "analog",
        "endconnectrules",
        "inout",
        "pow",
        "timer",
        "analysis",
        "enddiscipline",
        "input",
        "primitive",
        "tran",
        "and",
        "endfunction",
        "integer",
        "pull0",
        "tranif0",
        "asin",
        "endmodule",
        "join",
        "pull1",
        "tranif1",
        "asinh",
        "endnature",
        "laplace_nd",
        "pulldown",
        "transition",
        "assign",
        "endprimitive",
        "laplace_np",
        "pullup",
        "tri",
        "atan",
        "endspecify",
        "laplace_zd",
        "rcmos",
        "tri0",
        "atan2",
        "endtable",
        "laplace_zp",
        "real",
        "tri1",
        "atanh",
        "endtask",
        "large",
        "realtime",
        "triand",
        "begin",
        "event",
        "last_crossing",
        "reg",
        "trior",
        "branch",
        "exclude",
        "limexp",
        "release",
        "trireg",
        "buf",
        "exp",
        "ln",
        "repeat",
        "vectored",
        "bufif0",
        "final_step",
        "log",
        "rnmos",
        "wait",
        "bufif1",
        "flicker_noise",
        "macromodule",
        "rpmos",
        "wand",
        "case",
        "flow",
        "max",
        "rtran",
        "weak0",
        "casex",
        "for",
        "medium",
        "rtranif0",
        "weak1",
        "casez",
        "force",
        "min",
        "rtranif1",
        "while",
        "ceil",
        "forever",
        "module",
        "scalared",
        "white_noise",
        "cmos",
        "fork",
        "nand",
        "sin",
        "wire",
        "connectrules",
        "from",
        "nature",
        "sinh",
        "wor",
        "cos",
        "function",
        "negedge",
        "slew",
        "wreal",
        "cosh",
        "generate",
        "net_resolution",
        "small",
        "xnor",
        "cross",
        "genvar",
        "nmos",
        "specify",
        "xor",
        "ddt",
        "ground",
        "noise_table",
        "specparam",
        "zi_nd",
        "deassign",
        "highz0",
        "nor",
        "sqrt",
        "zi_np",
        "default",
        "highz1",
        "not",
        "strong0",
        "zi_zd",
        "defparam",
        "hypot",
        "notif0",
        "strong1",
        "zi_zp"
    )
)


def valid_keyword(name: str) -> bool:
    """check if name is a valid verilog-ams command.

    Parameters
    ----------
    name: `str`
        the command name

    Returns
    -------
    : `bool`
        valid keyword name.
    """
    return name.lower() in __keywords


def raise_exception_for_keyword(name: str):
    """Utility function to raise exception if keyword.

    Parameter
    ---------
    name: `str`
        name to make sure is not keywoard.
    """
    if valid_keyword(name):
        raise BaseException(f"{name} is not allowed to be keyword.")


class Keyword():
    """Keyword helps validate verilog-ams commands.
    """

    def __init__(self, name: str):
        """A verilog-ams keyword

        Parameters
        ----------
        name: `str`
            the name of the keyword.
        """
        name = name.lower()
        if not valid_keyword(name):
            raise BaseException(f"{name} is not a valid Verilog-AM/S command")
        self._name = name

    def __str__(self):
        return self._name
