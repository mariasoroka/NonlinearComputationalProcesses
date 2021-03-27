import numpy as np
def init_(prev_, current_, grid, h, tau, CFL, CFL_sign, number_of_x_steps, in_func):
    for i in range(0, number_of_x_steps):
        prev_[i] = in_func(grid[i])
        current_[i] = in_func(grid[i] - CFL_sign * CFL * h)
def real_solution(grid, h, tau, CFL, CFL_sign, number_of_time_steps, number_of_x_steps, in_func):
    real_sol = np.zeros(number_of_x_steps)
    for i in range(0, number_of_x_steps):
        real_sol[i] = in_func(grid[i] - CFL_sign * CFL * h * number_of_time_steps)
    return real_sol
def get_alphas(CFL, alpha00, alpha0_1):
    if(alpha00 == None or alpha0_1 == None):
        return np.array([None, None, None, None])
    alpha_10 = (2 - CFL - 2 * alpha00 - alpha0_1) / (2 + CFL)
    alpha0_2 = (2 * CFL - CFL * alpha00 - (CFL + 1) * alpha0_1) / (2 + CFL)
    return np.array([alpha00, alpha0_1, alpha0_2, alpha_10])
def solve(grid, h, tau, order, CFL, alpha00, alpha0_1, number_of_time_steps, number_of_x_steps, boundary_value, in_func):
    a = get_alphas(CFL, alpha00, alpha0_1)
    next_ = np.zeros(number_of_x_steps)
    current_ = np.zeros(number_of_x_steps)
    prev_ = np.zeros(number_of_x_steps)
    init_(prev_, current_, grid, h, tau, CFL, 1, number_of_x_steps, in_func)
    for i in range(0, number_of_time_steps):
        next_[0] = boundary_value
        if (order == 1):
            next_[1] = next_[0]
        elif (order == 2):
            next_[1] = current_[1] * (1 - CFL) + current_[0] * (CFL)
        elif (order == 3):
            next_[1] = current_[1] * (1 - CFL**2) + current_[2] * (CFL**2 / 2 - CFL / 2 ) + current_[0] * (CFL / 2 + CFL**2 / 2 )
        for j in range(2, number_of_x_steps):
            next_[j] = a[2] * current_[j - 2] + a[1] * current_[j - 1] + a[0] * current_[j] + a[3] * prev_[j]
        prev_ = current_.copy()
        current_ = next_.copy()
    return next_
def check_condition(tmp, a, b):
    if(a <= b):
        if(a <= tmp and tmp <= b):
            return 1
    else:
        if(b <= tmp and tmp <= a):
            return 1   
    return 0
def solve_hybrid(grid, h, tau, order, CFL, number_of_time_steps, number_of_x_steps, def_alpha00, def_alpha0_1, f_alpha00, f_alpha0_1, s_alpha00, s_alpha0_1, boundary_value, in_func):
    a = get_alphas(CFL, def_alpha00, def_alpha0_1)
    b = get_alphas(CFL, f_alpha00, f_alpha0_1)
    c = get_alphas(CFL, s_alpha00, s_alpha0_1)
    next_ = np.zeros(number_of_x_steps)
    current_ = np.zeros(number_of_x_steps)
    prev_ = np.zeros(number_of_x_steps)
    init_(prev_, current_, grid, h, tau, CFL, 1, number_of_x_steps, in_func)
    for i in range(0, number_of_time_steps):
        next_[0] = boundary_value
        if (order == 1):
            next_[1] = next_[0]
        elif (order == 2):
            next_[1] = current_[1] * (1 - CFL) + current_[0] * (CFL)
        elif (order == 3):
            next_[1] = current_[1] * (1 - CFL**2) + current_[2] * (CFL**2 / 2 - CFL / 2 ) + current_[0] * (CFL / 2 + CFL**2 / 2 )
        for j in range(2, number_of_x_steps):
            tmp = a[2] * current_[j - 2] + a[1] * current_[j - 1] + a[0] * current_[j] + a[3] * prev_[j]
            if (check_condition(tmp, current_[j], current_[j - 1])):
                next_[j] = tmp
            else:
                tmp = b[2] * current_[j - 2] + b[1] * current_[j - 1] + b[0] * current_[j] + b[3] * prev_[j]
                if (check_condition(tmp, current_[j], current_[j - 1]) or (s_alpha00 == None and s_alpha0_1 == None)):
                    next_[j] = tmp
                else:
                    next_[j] = c[2] * current_[j - 2] + c[1] * current_[j - 1] + c[0] * current_[j] + c[3] * prev_[j]
        prev_ = current_.copy()
        current_ = next_.copy()
    return next_
def solve_symmetric(grid, h, tau, order, CFL, alpha00, alpha0_1, number_of_time_steps, number_of_x_steps, boundary_value, in_func):
    a = get_alphas(CFL, alpha00, alpha0_1)
    next_ = np.zeros(number_of_x_steps)
    current_ = np.zeros(number_of_x_steps)
    prev_ = np.zeros(number_of_x_steps)
    init_(prev_, current_, grid, h, tau, CFL, -1, number_of_x_steps, in_func)
    for i in range(0, number_of_time_steps):
        next_[number_of_x_steps - 1] = boundary_value
        if (order == 1):
            next_[number_of_x_steps - 2] = next_[number_of_x_steps - 1]
        elif (order == 2):
            next_[number_of_x_steps - 2] = current_[number_of_x_steps - 2] * (1 - CFL) + current_[number_of_x_steps - 1] * (CFL)
        elif (order == 3):
            next_[number_of_x_steps - 2] = current_[number_of_x_steps - 2] * (1 - CFL**2) + current_[number_of_x_steps - 3] * (CFL**2 / 2 - CFL / 2 ) + current_[number_of_x_steps - 1] * (CFL / 2 + CFL**2 / 2 )
        for j in range(number_of_x_steps - 3, -1, -1):
            next_[j] = a[2] * current_[j + 2] + a[1] * current_[j + 1] + a[0] * current_[j] + a[3] * prev_[j]
        prev_ = current_.copy()
        current_ = next_.copy()
    return next_
def solve_hybrid_symmetric(grid, h, tau, order, CFL, number_of_time_steps, number_of_x_steps, def_alpha00, def_alpha0_1, f_alpha00, f_alpha0_1, s_alpha00, s_alpha0_1, boundary_value, in_func):
    a = get_alphas(CFL, def_alpha00, def_alpha0_1)
    b = get_alphas(CFL, f_alpha00, f_alpha0_1)
    c = get_alphas(CFL, s_alpha00, s_alpha0_1)
    next_ = np.zeros(number_of_x_steps)
    current_ = np.zeros(number_of_x_steps)
    prev_ = np.zeros(number_of_x_steps)
    init_(prev_, current_, grid, h, tau, CFL, 1, number_of_x_steps, in_func)
    for i in range(0, number_of_time_steps):
        next_[number_of_x_steps - 1] = boundary_value
        if (order == 1):
            next_[number_of_x_steps - 2] = next_[number_of_x_steps - 1]
        elif (order == 2):
            next_[number_of_x_steps - 2] = current_[number_of_x_steps - 2] * (1 - CFL) + current_[number_of_x_steps - 1] * (CFL)
        elif (order == 3):
            next_[number_of_x_steps - 2] = current_[number_of_x_steps - 2] * (1 - CFL**2) + current_[number_of_x_steps - 3] * (CFL**2 / 2 - CFL / 2 ) + current_[number_of_x_steps - 1] * (CFL / 2 + CFL**2 / 2 )
        for j in range(number_of_x_steps - 3, -1, -1):
            tmp = a[2] * current_[j + 2] + a[1] * current_[j + 1] + a[0] * current_[j] + a[3] * prev_[j]
            if (check_condition(tmp, current_[j], current_[j + 1])):
                next_[j] = tmp
            else:
                tmp = b[2] * current_[j + 2] + b[1] * current_[j + 1] + b[0] * current_[j] + b[3] * prev_[j]
                if (check_condition(tmp, current_[j], current_[j + 1]) or (s_alpha00 == None and s_alpha0_1 == None)):
                    next_[j] = tmp
                else:
                    next_[j] = c[2] * current_[j + 2] + c[1] * current_[j + 1] + c[0] * current_[j] + c[3] * prev_[j]
        prev_ = current_.copy()
        current_ = next_.copy()
    return next_