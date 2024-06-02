import numpy as np

def derivative(f, var):
    print("A-1.var.ndim ==", var.ndim)
    if var.ndim == 1:  # 1) b는 vector 이므로 1차원

        print("A-2.var ==", var)
        temp_var = var  #2) 원본 값 저장
        print("A-3.temp_var ==",temp_var)

        delta = 1e-5
        # 3) 미분계수 보관 변수 초기화
        diff_val = np.zeros(var.shape)
        print("A-4.diff_val ==",diff_val)

        # 4) 벡터의 모든 열(column) 순서대로 반복함
        for index in range(len(var)):
            print("A-5.index ==", index)
            print("A-6.len(var) ==", len(var))

            # 원본값에서 index 순서대로 할당
            target_var = float(temp_var[index])
            print("A-7.float(temp_var[index]) ==", float(temp_var[index]))
            print("A-8.target_var ==", target_var)

            temp_var[index] = target_var + delta
            print("A-9.temp_var[index] ==", temp_var[index])

            # x+delta 에 대한 함수 값 계산
            func_val_plus_delta = f(temp_var)
            print("A-10.func_val_plus_delta =",func_val_plus_delta)

            temp_var[index] = target_var - delta
            print("A-11.temp_var[index] ==", temp_var[index])

            # x-delta 에 대한 함수 값 계산
            func_val_minus_delta = f(temp_var)
            print("A-12.func_val_minus_delta ==", func_val_minus_delta)

            # 미분계수 계산 (도함수)
            diff_val[index] = (func_val_plus_delta - func_val_minus_delta) / \
                              (2 * delta)

            print("A-13.diff_val[index] (func_val_plus_delta - func_val_minus_delta) / (2 * delta)==", diff_val[index])

            # temp_var[index] 값이 delta에 의해 변경되어 변경전 값(target_var)을 할당
            temp_var[index] = target_var
            print("A-14.temp_var[index] ==", temp_var[index])

        return diff_val

    elif var.ndim == 2:  # matrix
        print("B-1.var.ndim ==", var.ndim)

        print("B-2.var ==", var)
        temp_var = var
        print("B-3.temp_var ==",temp_var)

        delta = 1e-5
        diff_val = np.zeros(var.shape)
        print("B-4.diff_val ==",diff_val)

        rows = var.shape[0]
        columns = var.shape[1]

        print("B-5.rows=", rows)
        print("B-6.columns =", columns)

        for row in range(rows):

            for column in range(columns):
                target_var = float(temp_var[row, column])
                print("B-7.float(temp_var[row, column]) ==", float(temp_var[row,column]))
                print("B-8.target_var ==", target_var)

                temp_var[row, column] = target_var + delta
                print("B-9.temp_var[row, column] ==", temp_var[row, column])

                # x+delta 에 대한 함수 값 계산
                func_val_plus_delta = f(temp_var)
                print("B-10.func_val_plus_delta =", func_val_plus_delta)

                temp_var[row, column] = target_var - delta
                print("B-11.temp_var[row, column] ==", temp_var[row, column])

                # x-delta 에 대한 함수 값 계산
                func_val_minus_delta = f(temp_var)
                print("B-10.func_val_minus_delta =", func_val_minus_delta)

                # 미분계수 계산 (도함수)
                diff_val[row, column] = \
                    (func_val_plus_delta - func_val_minus_delta) / (2 * delta)
                print("B-13.diff_val[row, column] (func_val_plus_delta - func_val_minus_delta) / (2 * delta) ==", diff_val[row, column])

                # temp_var[index] 값이 delta에 의해 변경되어 변경전 값(target_var)을 할당
                temp_var[row, column] = target_var
                print("B-14.temp_var[row, column] ==", temp_var[row, column])

        return diff_val

def func2(W):
     x = W[0]
     y = W[1]
     print("def_func2(W) x =", x, "y =", y)
     print("def_func2(W) 2*x + 3*x*y + np.power(y,3) =", 2*x + 3*x*y + np.power(y,3))
     return (2*x + 3*x*y + np.power(y,3))

f = lambda W : func2(W)

ret_val = derivative(f, np.array([1.0, 2.0]))
print(ret_val)
