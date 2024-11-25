# Absl Test or Non-Framework: A Slight Difference of PyTorch and Google Coding Style

When I left Google in 2010, Python was not as hot as it is now, probably because deep learning had not been something everyone talsk about.  However, I missed those *standard* libraries Google had for C++ programmers.  During the 12 years since then, I had been cheering for Google open-sourcing glog, gtest, and other parts of the standard libraries.  I have been imagining that some standard libraries must have been forging inside Google for Python, until I started reading some newer ex-Googler's code, which heavily relies on a package know as absl-py.

`absl-py` is a collect of frequently used Python packages that independently covers many use cases, and unit testing is one of them.  My previous work on PyTorch relies on pytest, whose design principle seems to be Occam's Razor.  An evidence is that `pytest` does not provide base classes like `unittest.TestCase`, which allows users to define methods like `setUp` and `tearDown` that will be called before and after the run of each test.  Instead, it depends on users to decide how to define these methods and how to call them.  Similarly, `pytest` does not provide parameterization of test parameters.  If you want it, you use a third-party library or write your own.   However, absl provides a testing framework that defines how `setUp` and `tearDown` should be defined and extends `unittest.TestCase` and adds capabilities of parametrization.

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
