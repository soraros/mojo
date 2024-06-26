# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from sys.info import bitwidthof

from testing import assert_equal


def test_properties():
    assert_equal(Int.MAX, (1 << bitwidthof[DType.index]() - 1) - 1)
    assert_equal(Int.MIN, -(1 << bitwidthof[DType.index]() - 1))


def test_add():
    assert_equal(Int.__add__(Int(3), Int(3)), 6)
    assert_equal(Int.__add__(Int(-2), Int(3)), 1)
    assert_equal(Int.__add__(Int(2), Int(-3)), -1)
    assert_equal(Int.__add__(Int(5), Int(-5)), 0)
    assert_equal(Int.__add__(Int(-5), Int(-4)), -9)


def test_sub():
    assert_equal(Int.__sub__(Int(3), Int(3)), 0)
    assert_equal(Int.__sub__(Int(-2), Int(3)), -5)
    assert_equal(Int.__sub__(Int(2), Int(-3)), 5)
    assert_equal(Int.__sub__(Int(5), Int(4)), 1)
    assert_equal(Int.__sub__(Int(4), Int(5)), -1)


def test_div():
    var n = Int(5)
    var d = Int(2)
    assert_equal(2.5, Int.__truediv__(n, d))
    Int.__itruediv__(n, d)
    assert_equal(2, n)


def test_pow():
    assert_equal(1, Int.__pow__(Int(3), Int(0)))
    assert_equal(27, Int.__pow__(Int(3), Int(3)))
    assert_equal(81, Int.__pow__(Int(3), Int(4)))


def test_ceil():
    assert_equal(Int.__ceil__(Int(5)), 5)
    assert_equal(Int.__ceil__(Int(0)), 0)
    assert_equal(Int.__ceil__(Int(-5)), -5)


def test_floor():
    assert_equal(Int.__floor__(Int(5)), 5)
    assert_equal(Int.__floor__(Int(0)), 0)
    assert_equal(Int.__floor__(Int(-5)), -5)


def test_round():
    assert_equal(Int.__round__(Int(5)), 5)
    assert_equal(Int.__round__(Int(0)), 0)
    assert_equal(Int.__round__(Int(-5)), -5)
    assert_equal(Int.__round__(5, 1), 5)
    assert_equal(Int.__round__(0, 1), 0)
    assert_equal(Int.__round__(-5, 1), -5)
    assert_equal(Int.__round__(100, -2), 100)


def test_trunc():
    assert_equal(Int.__trunc__(Int(5)), 5)
    assert_equal(Int.__trunc__(Int(0)), 0)
    assert_equal(Int.__trunc__(Int(-5)), -5)


def test_floordiv():
    assert_equal(1, Int.__floordiv__(Int(2), Int(2)))
    assert_equal(0, Int.__floordiv__(Int(2), Int(3)))
    assert_equal(-1, Int.__floordiv__(Int(2), Int(-2)))
    assert_equal(-50, Int.__floordiv__(Int(99), Int(-2)))
    assert_equal(-1, Int.__floordiv__(Int(-1), Int(10)))


def test_mod():
    assert_equal(0, Int.__mod__(Int(99), Int(1)))
    assert_equal(0, Int.__mod__(Int(99), Int(3)))
    assert_equal(-1, Int.__mod__(Int(99), Int(-2)))
    assert_equal(3, Int.__mod__(Int(99), Int(8)))
    assert_equal(-5, Int.__mod__(Int(99), Int(-8)))
    assert_equal(0, Int.__mod__(Int(2), Int(-1)))
    assert_equal(0, Int.__mod__(Int(2), Int(-2)))
    assert_equal(-1, Int.__mod__(Int(3), Int(-2)))
    assert_equal(1, Int.__mod__(Int(-3), Int(2)))


def test_divmod():
    var a: Int
    var b: Int
    a, b = divmod(7, 3)
    assert_equal(a, 2)
    assert_equal(b, 1)

    a, b = divmod(-7, 3)
    assert_equal(a, -3)
    assert_equal(b, 2)

    a, b = divmod(-7, -3)
    assert_equal(a, 2)
    assert_equal(b, -1)

    a, b = divmod(7, -3)
    assert_equal(a, -3)
    assert_equal(b, -2)

    a, b = divmod(0, 5)
    assert_equal(a, 0)
    assert_equal(b, 0)

    a, b = divmod(5, 0)
    assert_equal(a, 0)
    assert_equal(b, 0)


def test_abs():
    assert_equal(Int(-5).__abs__(), 5)
    assert_equal(Int(2).__abs__(), 2)
    assert_equal(Int(0).__abs__(), 0)


def test_string_conversion():
    assert_equal(Int(3).__str__(), "3")
    assert_equal(Int(-3).__str__(), "-3")
    assert_equal(Int(0).__str__(), "0")
    assert_equal(Int(100).__str__(), "100")
    assert_equal(Int(-100).__str__(), "-100")


def test_int_representation():
    assert_equal(Int(3).__repr__(), "3")
    assert_equal(Int(-3).__repr__(), "-3")
    assert_equal(Int(0).__repr__(), "0")
    assert_equal(Int(100).__repr__(), "100")
    assert_equal(Int(-100).__repr__(), "-100")


def test_indexer():
    assert_equal(5, Int(5).__index__())
    assert_equal(987, Int(987).__index__())


def main():
    test_properties()
    test_add()
    test_sub()
    test_div()
    test_pow()
    test_ceil()
    test_floor()
    test_round()
    test_trunc()
    test_floordiv()
    test_mod()
    test_divmod()
    test_abs()
    test_string_conversion()
    test_int_representation()
    test_indexer()
