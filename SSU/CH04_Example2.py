import numpy as np

def derivative(f, var):
    if var.ndim == 1:  # 1) b는 vector 이므로 1차원

        temp_var = var  #2) 원본 값 저장

        delta = 1e-5
        diff_val = np.zeros(var.shape)

        for index in range(len(var)):
            target_var = float(temp_var[index])

            temp_var[index] = target_var + delta
            func_val_plust_delta = f(temp_var)  # x+delta 에 대한 함수 값 계산
            temp_var[index] = target_var - delta
            func_val_minus_delta = f(temp_var)  # x-delta 에 대한 함수 값 계산
            diff_val[index] = (func_val_plust_delta - func_val_minus_delta) / (
                        2 * delta)

            temp_var[index] = target_var

        return diff_val

    elif var.ndim == 2:  # matrix

        temp_var = var

        delta = 1e-5
        diff_val = np.zeros(var.shape)

        rows = var.shape[0]
        columns = var.shape[1]

        for row in range(rows):

            for column in range(columns):
                target_var = float(temp_var[row, column])
                temp_var[row, column] = target_var + delta
                func_val_plus_delta = f(temp_var)  # x+delta 에 대한 함수 값 계산
                temp_var[row, column] = target_var - delta
                func_val_minus_delta = f(temp_var)  # x-delta 에 대한 함수 값 계산
                diff_val[row, column] = (func_val_plus_delta - func_val_minus_delta) / (
                                                    2 * delta)
                temp_var[row, column] = target_var

        return diff_val
