import numpy as np
import matplotlib.pyplot as plt
import math


def count_regression_matrix(matrix_x, a, b):
    return (a * matrix_x) + b


def regression_equation(matrix_x, matrix_y):
    matrix_factors = linear_regression_factors(matrix_x, matrix_y)
    if matrix_factors[1] < 0:
        return f"y = {matrix_factors[0]}x + ({matrix_factors[1]})"
    else:
        return f"y = {matrix_factors[0]}x + {matrix_factors[1]}"


def plot_title(pear_coef, a, b):
    if b < 0:
        return f"r={round(pear_coef, 2)}; y = {round(a, 1)}x + ({round(b, 1)})"
    else:
        return f"r={round(pear_coef, 2)}; y = {round(a, 1)}x + {round(b, 1)}"


def pearson_coefficient(matrix_x, matrix_y):
    return covariance(matrix_x, matrix_y) / (standard_deviation_m(matrix_x) * standard_deviation_m(matrix_y))


def linear_regression_factors(matrix_x, matrix_y):
    counted_covariance = covariance(matrix_x, matrix_y)
    matrix_x_variance = variance(matrix_x)
    a = counted_covariance / matrix_x_variance
    matrix_x_mean = mean_matrix(matrix_x)
    matrix_y_mean = mean_matrix(matrix_y)
    b = matrix_y_mean - (a * matrix_x_mean)
    return [a, b]


def covariance(matrix_x, matrix_y):
    if len(matrix_x) != len(matrix_y):
        return
    mean_x = mean_matrix(matrix_x)
    mean_y = mean_matrix(matrix_y)
    my_sum = 0
    for i in range(len(matrix_x)):
        my_sum += (matrix_x[i] - mean_x) * (matrix_y[i] - mean_y)
    return my_sum / (len(matrix_x) - 1)


def standard_deviation(matrix, col):
    matrix_col = matrix[:, col]
    return math.sqrt(variance(matrix_col))


def standard_deviation_m(matrix):
    return math.sqrt(variance(matrix))


def variance(matrix):
    counted_mean = mean_matrix(matrix)
    my_sum = 0.0
    for i in range(0, len(matrix)):
        my_sum += (matrix[i] - counted_mean) ** 2
    counted_variance = my_sum / (len(matrix) - 1)  # wzor dla proby (...)/(n-1)
    return counted_variance


def mean(matrix, col):
    matrix_col = matrix[:, col]
    return mean_matrix(matrix_col)


def mean_matrix(matrix):
    my_sum = 0.0
    for i in range(0, len(matrix)):
        my_sum += matrix[i]
    avg = (my_sum / len(matrix))
    return avg


iris = np.loadtxt('data.csv', delimiter=',')

sepal_length = iris[:, 0]
sepal_width = iris[:, 1]
petal_length = iris[:, 2]
petal_width = iris[:, 3]

# Współczynniki Pearsona:
print("Współczynniki Pearsona: ")

sepal_length_to_sepal_width_pear_coef = pearson_coefficient(sepal_length, sepal_width)
print("Wykres 1:", round(sepal_length_to_sepal_width_pear_coef, 2))

sepal_length_to_petal_length_pear_coef = pearson_coefficient(sepal_length, petal_length)
print("Wykres 2:", round(sepal_length_to_petal_length_pear_coef, 2))

sepal_length_to_petal_width_pear_coef = pearson_coefficient(sepal_length, petal_width)
print("Wykres 3:", round(sepal_length_to_petal_width_pear_coef, 2))

sepal_width_to_petal_length_pear_coef = pearson_coefficient(sepal_width, petal_length)
print("Wykres 4:", round(sepal_width_to_petal_length_pear_coef, 2))

sepal_width_to_petal_width_pear_coef = pearson_coefficient(sepal_width, petal_width)
print("Wykres 5:", round(sepal_width_to_petal_width_pear_coef, 2))

petal_length_to_petal_width_pear_coef = pearson_coefficient(petal_length, petal_width)
print("Wykres 6:", round(petal_length_to_petal_width_pear_coef, 2))

print('\n')

# Równania Regresji Liniowej:
print("Równania regresji liniowej: ")

print("Wykres 1: ", regression_equation(sepal_length, sepal_width))

print("Wykres 2: ", regression_equation(sepal_length, petal_length))

print("Wykres 3: ", regression_equation(sepal_length, petal_width))

print("Wykres 4: ", regression_equation(sepal_width, petal_length))

print("Wykres 5: ", regression_equation(sepal_width, petal_width))

print("Wykres 6: ", regression_equation(petal_length, petal_width))

# Wykresy:
figure, axis = plt.subplots(3, 2)

# wykres_1
chart_1_factors = linear_regression_factors(sepal_length, sepal_width)
a_1, b_1 = chart_1_factors[0], chart_1_factors[1]
y = count_regression_matrix(sepal_length, a_1, b_1)
axis[0, 0].plot(sepal_length, y, color="red")

axis[0, 0].set_title(plot_title(sepal_length_to_sepal_width_pear_coef, chart_1_factors[0], chart_1_factors[1]))
axis[0, 0].set_xlabel("Długość działki kielicha (cm)")
axis[0, 0].set_ylabel("Szerokość działki kielicha (cm)")
axis[0, 0].scatter(sepal_length, sepal_width)

# wykres_2
chart_2_factors = linear_regression_factors(sepal_length, petal_length)
a_2, b_2 = chart_2_factors[0], chart_2_factors[1]
y = count_regression_matrix(sepal_length, a_2, b_2)
axis[0, 1].plot(sepal_length, y, color="red")

axis[0, 1].set_title(plot_title(sepal_length_to_petal_length_pear_coef, chart_2_factors[0], chart_2_factors[1]))
axis[0, 1].set_xlabel("Długość działki kielicha (cm)")
axis[0, 1].set_ylabel("Długość płatka (cm)")
axis[0, 1].scatter(sepal_length, petal_length)

# wykres_3
chart_3_factors = linear_regression_factors(sepal_length, petal_width)
a_3, b_3 = chart_3_factors[0], chart_3_factors[1]
y = count_regression_matrix(sepal_length, a_3, b_3)
axis[1, 0].plot(sepal_length, y, color="red")

axis[1, 0].set_title(plot_title(sepal_length_to_petal_width_pear_coef, chart_3_factors[0], chart_3_factors[1]))

axis[1, 0].set_xlabel("Długość działki kielicha (cm)")
axis[1, 0].set_ylabel("Szerokość płatka (cm)")
axis[1, 0].scatter(sepal_length, petal_width)

# wykres_4
chart_4_factors = linear_regression_factors(sepal_width, petal_length)
a_4, b_4 = chart_4_factors[0], chart_4_factors[1]
y = count_regression_matrix(sepal_width, a_4, b_4)
axis[1, 1].plot(sepal_width, y, color="red")

axis[1, 1].set_title(plot_title(sepal_width_to_petal_length_pear_coef, chart_4_factors[0], chart_4_factors[1]))

axis[1, 1].set_xlabel("Szerokość działki kielicha (cm)")
axis[1, 1].set_ylabel("Długość płatka (cm)")
axis[1, 1].scatter(sepal_width, petal_length)

# wykres_5
chart_5_factors = linear_regression_factors(sepal_width, petal_width)
a_5, b_5 = chart_5_factors[0], chart_5_factors[1]
y = count_regression_matrix(sepal_width, a_5, b_5)
axis[2, 0].plot(sepal_width, y, color="red")

axis[2, 0].set_title(plot_title(sepal_width_to_petal_width_pear_coef, chart_5_factors[0], chart_5_factors[1]))
axis[2, 0].set_xlabel("Szerokość działki kielicha (cm)")
axis[2, 0].set_ylabel("Szerokość płatka (cm)")
axis[2, 0].scatter(sepal_width, petal_width)

# wykres_6
chart_6_factors = linear_regression_factors(petal_length, petal_width)
a_6, b_6 = chart_6_factors[0], chart_6_factors[1]
y = count_regression_matrix(petal_length, a_6, b_6)
axis[2, 1].plot(petal_length, y, color="red")

axis[2, 1].set_title(plot_title(petal_length_to_petal_width_pear_coef, chart_6_factors[0], chart_6_factors[1]))
axis[2, 1].set_xlabel("Długość płatka (cm)")
axis[2, 1].set_ylabel("Szerokość płatka (cm)")
axis[2, 1].scatter(petal_length, petal_width)
figure.set_size_inches(14, 16)
plt.savefig("./charts/all_in_one.png")
plt.show()

# # wykres_1
# chart_1_factors = linear_regression_factors(sepal_length, sepal_width)
# a_1, b_1 = chart_1_factors[0], chart_1_factors[1]
# y = count_regression_matrix(sepal_length, a_1, b_1)
# plt.plot(sepal_length, y, color="red")
#
# plt.title(plot_title(sepal_length_to_sepal_width_pear_coef, chart_1_factors[0], chart_1_factors[1]))
# plt.xlabel("Długość działki kielicha (cm)")
# plt.ylabel("Szerokość działki kielicha (cm)")
# plt.scatter(sepal_length, sepal_width)
# plt.savefig("./charts/szer_dkiel_dłg_dkiel.png")
# plt.show()
#
# # wykres_2
# chart_2_factors = linear_regression_factors(sepal_length, petal_length)
# a_2, b_2 = chart_2_factors[0], chart_2_factors[1]
# y = count_regression_matrix(sepal_length, a_2, b_2)
# plt.plot(sepal_length, y, color="red")
#
# plt.title(plot_title(sepal_length_to_petal_length_pear_coef, chart_2_factors[0], chart_2_factors[1]))
# plt.xlabel("Długość działki kielicha (cm)")
# plt.ylabel("Długość płatka (cm)")
# plt.scatter(sepal_length, petal_length)
# plt.savefig("./charts/dłg_płt_dłg_dkiel.png")
# plt.show()
#
# # wykres_3
# chart_3_factors = linear_regression_factors(sepal_length, petal_width)
# a_3, b_3 = chart_3_factors[0], chart_3_factors[1]
# y = count_regression_matrix(sepal_length, a_3, b_3)
# plt.plot(sepal_length, y, color="red")
#
# plt.title(plot_title(sepal_length_to_petal_width_pear_coef, chart_3_factors[0], chart_3_factors[1]))
#
# plt.xlabel("Długość działki kielicha (cm)")
# plt.ylabel("Szerokość płatka (cm)")
# plt.scatter(sepal_length, petal_width)
# plt.savefig("./charts/szer_płt_dłg_dkiel.png")
# plt.show()
#
# # wykres_4
# chart_4_factors = linear_regression_factors(sepal_width, petal_length)
# a_4, b_4 = chart_4_factors[0], chart_4_factors[1]
# y = count_regression_matrix(sepal_width, a_4, b_4)
# plt.plot(sepal_width, y, color="red")
#
# plt.title(plot_title(sepal_width_to_petal_length_pear_coef, chart_4_factors[0], chart_4_factors[1]))
#
# plt.xlabel("Szerokość działki kielicha (cm)")
# plt.ylabel("Długość płatka (cm)")
# plt.scatter(sepal_width, petal_length)
# plt.savefig("./charts/dłg_płt_szer_dkiel.png")
# plt.show()
#
# # wykres_5
# chart_5_factors = linear_regression_factors(sepal_width, petal_width)
# a_5, b_5 = chart_5_factors[0], chart_5_factors[1]
# y = count_regression_matrix(sepal_width, a_5, b_5)
# plt.plot(sepal_width, y, color="red")
#
# plt.title(plot_title(sepal_width_to_petal_width_pear_coef, chart_5_factors[0], chart_5_factors[1]))
# plt.xlabel("Szerokość działki kielicha (cm)")
# plt.ylabel("Szerokość płatka (cm)")
# plt.scatter(sepal_width, petal_width)
# plt.savefig("./charts/szer_płt_szer_dkiel.png")
# plt.show()
#
# # wykres_6
# chart_6_factors = linear_regression_factors(petal_length, petal_width)
# a_6, b_6 = chart_6_factors[0], chart_6_factors[1]
# y = count_regression_matrix(petal_length, a_6, b_6)
# plt.plot(petal_length, y, color="red")
#
# plt.title(plot_title(petal_length_to_petal_width_pear_coef, chart_6_factors[0], chart_6_factors[1]))
# plt.xlabel("Długość płatka (cm)")
# plt.ylabel("Szerokość płatka (cm)")
# plt.scatter(petal_length, petal_width)
# plt.savefig("./charts/szer_płt_dłg_płt.png")
# plt.show()
