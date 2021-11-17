import cbadc.verilog_ams.keywords as keywords


def test_valid_keyword():
    assert(keywords.valid_keyword('if'))
    assert(keywords.valid_keyword('analog'))
    assert(keywords.valid_keyword('begin'))
    assert(keywords.valid_keyword('branch'))
    assert(not keywords.valid_keyword(' if'))
    assert(not keywords.valid_keyword(' whatever'))


def test_keyword_class():
    key = keywords.Keyword('BeGin')
    print(key)
    assert(str(key) == "begin")
