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


class MemoryManagement(unittest.TestCase):
    def test_weakref_output(self):
        for i in range(10):
            x = Variable(np.random.randn(10000))  # huge data
            y = square(square(square(x)))  # complicated calculation

    def test_deletion_interim_grad(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()

        self.assertEqual(y.grad, None)
        self.assertEqual(t.grad, None)
        self.assertEqual(x0.grad, 2.0)
        self.assertEqual(x1.grad, 1.0)

    def test_enable_backprop(self):
        with no_grad():
            x = Variable(np.array(2.0))
            y = square(x)


class OperationOverload(unittest.TestCase):
    def test_add_mul_operation(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        # y =add(mul(a, b), c)
        y = a * b + c
        y.backward()

        self.assertEqual(y.data, np.array(7.0))
        self.assertEqual(a.grad, np.array(2.0))
        self.assertEqual(b.grad, np.array(3.0))

    def test_operation_variable_and_ndarray(self):
        x = Variable(np.array(2.0))
        y = x + np.array(3.0)
        self.assertEqual(y.data, np.array(5.0))
        z = np.array([2.0]) + x
        self.assertEqual(z.data, np.array(4.0))

    def test_operation_variable_and_number(self):
        x = Variable(np.array(2.0))
        y = x + 3.0
        self.assertEqual(y.data, np.array(5.0))
        z = 3.0 * x + 1.0
        self.assertEqual(z.data, np.array(7.0))

    def test_neg(self):
        x = Variable(np.array(2.0))
        y = -x
        self.assertEqual(y.data, -2.0)

    def test_sub(self):
        x = Variable(np.array(2.0))
        y1 = 2.0 - x
        y2 = x - 1.0
        self.assertEqual(y1.data, 0.0)
        self.assertEqual(y2.data, 1.0)

    def test_div(self):
        x = Variable(np.array(3.0))
        y1 = x / 2.0
        y2 = 6.0 / x
        self.assertEqual(y1.data, 1.5)
        self.assertEqual(y2.data, 2.0)