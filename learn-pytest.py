from pytest.mark import parametrize


def setUp():
    print("setUp")


def tearDown():
    print("tearDown")


@parametrize(
    "x, y, z", [(1, 2, 3), (4, 5, 9)]
)
def test_a(x, y, z):
    setUp()
    print("test_a", x, y, z)
    tearDown()


def test_b():
    setUp()
    print("test_b")
    tearDown()
