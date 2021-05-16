from step import *
import unittest


class SquareTest(unittest.TestCase):
    # def test_forward(self):
    #     x = Variable(np.array(2.0))
    #     y = square(x)
    #     excepted = np.array(4.0)
    #     self.assertEqual(y.data, excepted)

    # def test_backward(self):
    #     x = Variable(np.array(3.0))
    #     y = square(x)
    #     y.backward()
    #     expected = np.array(6.0)
    #     self.assertEqual(x.grad, expected)

    # def test_gradient_check(self):
    #     x = Variable(np.random.rand(1))  # generate random value
    #     y = square(x)
    #     y.backward()
    #     num_grad = numerical_diff(square, x)
    #     # determine whether two arguments are close or not
    #     flg = np.allclose(x.grad, num_grad)
    #     self.assertTrue(flg)

    def test_add_square(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))

        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(13.0, z.data)
        self.assertEqual(4.0, x.grad)
        self.assertEqual(6.0, y.grad)


class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        expected = np.array(5.0)
        self.assertEqual(y.data, expected)

    def test_add_same_variables(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        z = add(add(x, x), x)
        self.assertEqual(y.data, np.array(6.0))
        y.backward()
        self.assertEqual(x.grad, np.array(2.0))
        x.cleargrad()
        z.backward()
        self.assertEqual(x.grad, np.array(3.0))


class BackwardTest(unittest.TestCase):
    def test_generation_sort(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()

        self.assertEqual(y.data, np.array(32.0))
        self.assertEqual(x.grad, np.array(64.0))
