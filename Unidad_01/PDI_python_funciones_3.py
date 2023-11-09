def mi_funcion(x, y):
    out_add = x + y
    out_prod = x*y
    out_pow = x**y
    print(f"Add: {out_add}")
    print(f"prod: {out_prod}")
    print(f"pow: {out_pow}")
    return out_add, out_prod, out_pow

out = mi_funcion(1,2)
out_1, out_2, out_3 = mi_funcion(1,2)
out_1, out_2, _ = mi_funcion(1,2)
_, out_2, _ = mi_funcion(1,2)
out_1, _, _ = mi_funcion(1,2)
out_1, *_ = mi_funcion(1,2)
*_, out_3 = mi_funcion(1,2)

out_1, _ = mi_funcion(1,2)  # Error
_, out_2 = mi_funcion(1,2)  # Error


