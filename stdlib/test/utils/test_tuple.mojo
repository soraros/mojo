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

from testing import assert_equal, assert_false, assert_true

from memory import UnsafePointer
from utils import IndexList, StaticTuple
from test_utils import ValueDestructorRecorder


def test_static_tuple():
    var tup1 = StaticTuple[Int, 1](1)
    assert_equal(tup1[0], 1)

    var tup2 = StaticTuple[Int, 2](1, 1)
    assert_equal(tup2[0], 1)
    assert_equal(tup2[1], 1)

    var tup3 = StaticTuple[Int, 3](1, 2, 3)
    assert_equal(tup3[0], 1)
    assert_equal(tup3[1], 2)
    assert_equal(tup3[2], 3)

    assert_equal(tup3[0], 1)
    assert_equal(tup3[Int(0)], 1)


def test_static_int_tuple():
    assert_equal(str(IndexList[1](1)), "(1,)")

    assert_equal(str(IndexList[3](2)), "(2, 2, 2)")

    assert_equal(
        str(IndexList[3](1, 2, 3) * IndexList[3](4, 5, 6)),
        "(4, 10, 18)",
    )

    assert_equal(
        str(IndexList[4](1, 2, 3, 4) - IndexList[4](4, 5, 6, 7)),
        "(-3, -3, -3, -3)",
    )

    assert_equal(str(IndexList[2](10, 11) // IndexList[2](3, 4)), "(3, 2)")

    # Note: index comparison is intended for access bound checking, which is
    #  usually all-element semantic, i.e. true if true for all positions.
    assert_true(IndexList[5](1, 2, 3, 4, 5) < IndexList[5](4, 5, 6, 7, 8))

    assert_false(IndexList[4](3, 5, -1, -2) > IndexList[4](0, 0, 0, 0))

    assert_equal(len(IndexList[4](3, 5, -1, -2)), 4)

    assert_equal(str(IndexList[2]((1, 2))), "(1, 2)")

    assert_equal(str(IndexList[4]((1, 2, 3, 4))), "(1, 2, 3, 4)")


def test_tuple_literal():
    assert_equal(len((1, 2, (3, 4), 5)), 4)
    assert_equal(len(()), 0)


def main():
    test_static_tuple()
    test_static_int_tuple()
    test_tuple_literal()
