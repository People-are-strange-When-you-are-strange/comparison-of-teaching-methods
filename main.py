import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split


# максимальные итерации
def gradient_descent(max_iterations, threshold, w_init, obj_func, grad_func, extra_param=None, learning_rate=0.05, momentum=0.8):
    #избегание ошибки класса
    if extra_param is None:
        extra_param = []
    w = w_init
    w_history = w
    f_history = obj_func(w, extra_param)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10

    while i < max_iterations and diff > threshold:
        delta_w = -learning_rate * grad_func(w, extra_param) + momentum * delta_w
        w = w + delta_w

        # store the history of w and f
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, extra_param)))

        #обновить номер итерации и diff между последовательными значениями целевой функции
        i += 1
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


def visualize_fw():
    xcoord = np.linspace(-10.0, 10.0, 50)
    ycoord = np.linspace(-10.0, 10.0, 50)
    w1, w2 = np.meshgrid(xcoord, ycoord)
    #сжимание измерения до 1-го
    pts = np.vstack((w1.flatten(), w2.flatten()))

    #ВСЕ 2D в гриде
    pts = pts.transpose()

    # Значения функции в каждой точке
    f_vals = np.sum(pts * pts, axis=1)
    function_plot(pts, f_vals)
    #   plt.title('Objective Function Shown in Color')
    #   plt.show()
    return pts, f_vals


def annotate_pt(text, xy, xytext, color):
    plt.plot(xy[0], xy[1], marker='P', markersize=10, c=color)
    plt.annotate(text, xy=xy, xytext=xytext,
                 # color=color,
                 arrowprops=dict(arrowstyle="->",
                                 color=color,
                                 connectionstyle='arc3'))


# Plot the function
# Pts — это 2D-точки, а f_val — соответствующее значение функции
def function_plot(pts, f_val):
    f_plot = plt.scatter(pts[:, 0], pts[:, 1],
                         c=f_val, vmin=min(f_val), vmax=max(f_val),
                         cmap='RdBu_r')
    plt.colorbar(f_plot)
    # Show the optimal point
    annotate_pt('global minimum', (0, 0), (-5, -7), 'yellow')


pts, f_vals = visualize_fw()


# Objective function
def f(w, extra=[]):
    return np.sum(w * w)


# Function to compute the gradient
#ignor problem [] (невозможны проблемы клас.)
def grad(w, extra=[]):
    return 2 * w


# Функция для построения целевой функции
# и история изучения отмечена стрелками
# чтобы показать, как проходило обучение
def visualize_learning(w_history):
    # Make the function plot
    function_plot(pts, f_vals)

    # Plot the history
    plt.plot(w_history[:, 0], w_history[:, 1], marker='o', c='magenta')

    # Annotate the point found at last iteration
    annotate_pt('minimum found', (w_history[-1, 0], w_history[-1, 1]), (-1, 7), 'green')
    iter = w_history.shape[0]
    for w, i in zip(w_history, range(iter - 1)):
        # Annotate with arrows to show history
        plt.annotate("",
                     xy=w, xycoords='data',
                     xytext=w_history[i + 1, :], textcoords='data',
                     arrowprops=dict(arrowstyle='<-',
                                     connectionstyle='angle3'))


def solve_fw():
    # Setting up
    rand = np.random.RandomState(19)
    w_init = rand.uniform(-10, 10, 2)
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(18, 12))
    learning_rates = [0.05, 0.2, 0.5, 0.8]
    momentum = [0, 0.5, 0.99]
    ind = 1

    # Полный перебор алеф-параметров
    for alpha in momentum:
        for eta, col in zip(learning_rates, [0, 1, 2, 3]):
            plt.subplot(3, 4, ind)
            w_history, f_history = gradient_descent(5, -1, w_init, f, grad, [], eta, alpha)

            visualize_learning(w_history)
            ind = ind + 1
            plt.text(-9, 12, 'Learning Rate = ' + str(eta), fontsize=13)
            if col == 1:
                plt.text(10, 15, 'momentum = ' + str(alpha), fontsize=20)

    fig.subplots_adjust(hspace=0.5, wspace=.3)
    plt.show()


# Входной аргумент — это вес и кортеж (train_data, target)
def grad_mse(w, xy):
    (x, y) = xy
    (rows, cols) = x.shape

    # Compute the output
    o = np.sum(x * w, axis=1)
    diff = y - o
    diff = diff.reshape((rows, 1))
    diff = np.tile(diff, (1, cols))
    grad = diff * x
    grad = -np.sum(grad, axis=0)
    return grad


# Input argument is weight and a tuple (train_data, target)
def mse(w, xy):
    (x, y) = xy

#Выходные данные # !!!ИСПОЛЬЗУЮ mse, а не mse/m # поэтому это будет иметь отношение к конечному результату
    o = np.sum(x * w, axis=1)
    mse = np.sum((y - o) * (y - o))
    mse = mse / 2
    return mse

# Загружаем набор данных с двумя классами
digits,target = dt.load_digits(n_class=2,return_X_y=True)
fig,ax = plt.subplots(nrows=1, ncols=10,figsize=(12,4),subplot_kw=dict(xticks=[], yticks=[]))

# Рисунки нескольких изображений цифр
for i in np.arange(10):
    ax[i].imshow(digits[i,:].reshape(8,8),cmap=plt.cm.gray)
plt.show()

# Разделить на обучающий и тестовый наборы
x_train, x_test, y_train, y_test = train_test_split(
                        digits, target, test_size=0.2, random_state=10)

# Добавьте столбец из единиц для учета систематической ошибки в обучении и тестировании
#Переписать через???? функцию nympy????
x_train = np.hstack((np.ones((y_train.size,1)),x_train))
x_test  = np.hstack((np.ones((y_test.size,1)),x_test))

# Инициализируем веса и вызываем градиентный спуск
rand = np.random.RandomState(19)
w_init = rand.uniform(-1,1,x_train.shape[1])*.000001
w_history,mse_history = gradient_descent(100,0.1,w_init,
                              mse,grad_mse,(x_train,y_train),
                             learning_rate=1e-6,momentum=0.7)

# Plot the MSE
plt.plot(np.arange(mse_history.size),mse_history)
plt.xlabel('Iteration No.')
plt.ylabel('Mean Square Error')
plt.title('Gradient Descent on Digits Data (Batch Version)')
plt.legend(['grad', 'batch'])

# Возвращает частоту ошибок классификатора
# все ошибочных классификаций/все*100


def error(w, xy):
    (x, y) = xy
    o = np.sum(x * w, axis=1)

    # сопоставить выходные значения с метками класса 0/1
    ind_1 = np.where(o > 0.5)
    ind_0 = np.where(o <= 0.5)
    o[ind_1] = 1
    o[ind_0] = 0
    return np.sum((o - y) * (o - y)) / y.size * 100


train_error = error(w_history[-1], (x_train, y_train))
test_error = error(w_history[-1], (x_test, y_test))


# (xy) is the (training_set,target) pair
def stochastic_gradient_descent(max_epochs, threshold, w_init,
                                obj_func, grad_func, xy,
                                learning_rate=0.05, momentum=0.8):
    (x_train, y_train) = xy
    w = w_init
    w_history = w
    f_history = obj_func(w, xy)
    delta_w = np.zeros(w.shape)
    i = 0
    diff = 1.0e10
    rows = x_train.shape[0]

    # Run epochs
    while i < max_epochs and diff > threshold:
        # Перемешать строки, используя фиксированное начальное число, чтобы воспроизвести результаты
        np.random.seed(i)
        p = np.random.permutation(rows)

        # Запуск для каждого экземпляра/примера в тренировочном наборе
        for x, y in zip(x_train[p, :], y_train[p]):
            delta_w = -learning_rate * grad_func(w, (np.array([x]), y)) + momentum * delta_w
            w = w + delta_w

        i += 1
        w_history = np.vstack((w_history, w))
        f_history = np.vstack((f_history, obj_func(w, xy)))
        diff = np.absolute(f_history[-1] - f_history[-2])

    return w_history, f_history


rand = np.random.RandomState(19)
w_init = rand.uniform(-1,1,x_train.shape[1])*.000001
w_history_stoch,mse_history_stoch = stochastic_gradient_descent(
                                100,0.1,w_init,
                              mse,grad_mse,(x_train,y_train),
                             learning_rate=1e-6,momentum=0.7)

# Plot the MSE
plt.plot(np.arange(mse_history_stoch.size),mse_history_stoch)
plt.xlabel('Iteration No.')
plt.ylabel('Mean Square Error')
plt.title('Gradient Descent on Digits Data (Stochastic Version)')


fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 3))

rand = np.random.RandomState(11)
w_init = rand.uniform(-1, 1, x_train.shape[1]) * .000001
eta = 1e-6
for alpha, ind in zip([0, 0.5, 0.9], [1, 2, 3]):
    w_history, mse_history = gradient_descent(
        100, 0.01, w_init,
        mse, grad_mse, (x_train, y_train),
        learning_rate=eta, momentum=alpha)

    w_history_stoch, mse_history_stoch = stochastic_gradient_descent(
        100, 0.01, w_init,
        mse, grad_mse, (x_train, y_train),
        learning_rate=eta, momentum=alpha)

    # Plot the MSE
    plt.subplot(130 + ind)
    plt.plot(np.arange(mse_history.size), mse_history, color='green')
    plt.plot(np.arange(mse_history_stoch.size), mse_history_stoch, color='blue')
    plt.legend(['batch', 'stochastic'])

    # Display total iterations
    plt.text(3, -30, 'Batch: Iterations=' +
             str(mse_history.size))
    plt.text(3, -45, 'Stochastic: Iterations=' +
             str(mse_history_stoch.size))
    plt.title('Momentum = ' + str(alpha))

    # Display the error rates
    train_error = error(w_history[-1], (x_train, y_train))
    test_error = error(w_history[-1], (x_test, y_test))

    train_error_stochastic = error(w_history_stoch[-1], (x_train, y_train))
    test_error_stochastic = error(w_history_stoch[-1], (x_test, y_test))

    print('Momentum = ' + str(alpha))

    print('\tПакетный градиентный спуск:')
    print('\t\tTrain error: ' + "{:.2f}".format(train_error))
    print('\t\tTest error: ' + "{:.2f}".format(test_error))

    print('\tСтохастический градиентный спуск :')
    print('\t\tTrain error: ' + "{:.2f}".format(train_error_stochastic))
    print('\t\tTest error: ' + "{:.2f}".format(test_error_stochastic))

plt.show()