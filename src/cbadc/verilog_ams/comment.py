"""Tools for generating verilog am/s styled comments.
"""
from .string_literals import tab, new_line


class Comment:
    """Generate a verilog-ams comment.
    """

    def __init__(self, comment: str):
        """Instantiate the comment

        Parameters
        ----------
        comment: `str`
            a text string holding the verilog-ams comment.
        """
        self._command = "// "
        self._row_length = 90 - len(tab) - len(self._command)
        temp = comment.split(" ")
        column_count = 0
        self._comment = [temp[0] + " "]
        for w in temp[1:]:
            if len(w) + column_count + 1 < self._row_length:
                self._comment[-1] += w + " "
                column_count += len(w) + 1
            else:
                self._comment.append(w + " ")
                column_count = len(w)

        # Remove trailing whitespaces
        for index, row in enumerate(self._comment):
            self._comment[index] = row.strip()

    def __str__(self) -> str:
        start = tab + self._command
        return start + (new_line + start).join(self._comment)
