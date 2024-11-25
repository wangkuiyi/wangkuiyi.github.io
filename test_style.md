# Unit Test Library Or Framework?

When I left Google in 2010, I missed the standard libraries that Google had for C++ programmers. I was happy when Google open-sourced glog, gtest, and other libraries. Recently, I learned absl-py, a collection of frequently used Python libraries that Google has open-sourced, by reading the code of some newer ex-Googlers.

Part of absl-py focuses on unit testing. My previous work involved PyTorch, which utilized pytest, a minimalist Python unit test library, in contrast to Python’s standard unittest framework. Unlike `unittest`, which provides the base class `unittest.TestCase` that necessitates users to define methods like `setUp` and `tearDown` and automatically calls them before and after each test execution, `pytest` lacks such a base class. Instead, it relies on users to define these methods and manually call them. While `pytest` offers parameterization of test parameters, it also provides the flexibility to incorporate third-party options. On the other hand, absl-py provides a framework that extends `unittest.TestCase`.

```
from pytest.mark import parametrize        from absl.testing.parameterized import (
                                               TestCase,
                                               parameters,
def setUp():                               )
    print("setUp")

                                           class MyTestCase(TestCase):
def tearDown():                                def setUp(self):
    print("tearDown")                              print("setup")

                                               def tearDown(self):
@parametrize(                                      print("tearDown")
    "x, y, z", [(1, 2, 3), (4, 5, 9)]
)                                              @parameters((1, 2, 3), (4, 5, 9))
def test_a(x, y, z):                           def test_a(self, a, b, c):
    setUp()                                        print("test_a", a, b, c)
    print("test_a", x, y, z)
    tearDown()                                 def test_b(self):
                                                   print("test_b")
def test_b():
    setUp()
    print("test_b")
    tearDown()
```
