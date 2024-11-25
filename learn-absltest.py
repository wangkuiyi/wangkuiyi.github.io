from absl.testing.parameterized import (
    TestCase,
    parameters,
)


class MyTestCase(TestCase):
    def setUp(self):
        print("setup")

    def tearDown(self):
        print("tearDown")

    @parameters((1, 2, 3), (4, 5, 9))
    def test_a(self, a, b, c):
        print("test_a", a, b, c)

    def test_b(self):
        print("test_b")
