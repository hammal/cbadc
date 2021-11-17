import cbadc.verilog_ams.comment as comment


def test_initialization():
    text = "Just a comment."
    c = comment.Comment(text)
    print(c, text)
    assert(str(c) == '\t// ' + text)


def test_a_row_breaking_comment():
    c = comment.Comment(
        'Just a very long comment that should span two rows at the very least. Right? Perhaps there would be even more text needed? Could I make it stretch three rows? That would make a nice test')
    print(c)
    expected_string = "\t// Just a very long comment that should span two rows at the very least. Right? Perhaps\n\t// there would be even more text needed? Could I make it stretch three rows? That would\n\t// make a nice test"
    assert(str(c) == expected_string)
