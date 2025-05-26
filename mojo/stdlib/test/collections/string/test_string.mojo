from testing import assert_equal, assert_raises


def main():
    with assert_raises(contains=String("")):
        _ = String("{}").format()

    assert_equal(String("}}").format(), "}")
    assert_equal(String("{{").format(), "{")
