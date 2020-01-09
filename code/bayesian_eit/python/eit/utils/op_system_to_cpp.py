func_template = """
Eigen::MatrixXd {name} (const Eigen::MatrixXd &xarg, const Eigen::MatrixXd &yarg, const Eigen::VectorXd &args) {{
    {extract_args}
    
    Eigen::MatrixXd ret(x.rows(), y.rows());
    for(int i = 0; i < x.rows(); i++) {{
        {extract_x_params}
        for(int j = 0; j < y.rows(); j++) {{
            {extract_y_params}
            
            ret(i, j) = {code};
        }}
    }}
    return ret;
}}
"""

assign_template = """double {} = {};"""

# This was originally used to create operators.cpp in cpp/src 
# NB this produces highly inefficient code, which was then optimised by hand.
def construct_cpp_operators(kernel, names, names_bar, ops, ops_bar, symbols):
    extract_args = ""
    extract_x_params = ""
    extract_y_params = ""
    
    for i, s in enumerate(symbols[0]):
        extract_x_params += assign_template.format(s.name, "xarg(i, {})".format(i)) + "\n"
    for i, s in enumerate(symbols[1]):
        extract_y_params += assign_template.format(s.name, "yarg(j, {})".format(i)) + "\n"
    for i, s in enumerate(symbols[2]):
        extract_args += assign_template.format(s.name, "args({})".format(i)) + "\n"
    funcs = []
    
    for i in xrange(len(ops)):
        for j in xrange(i, len(ops)):
            op = ops[i]
            op_bar = ops_bar[j]
            name = names[i] + '_' + names[j]

            expr = op(op_bar(kernel))
            code = sp.printing.ccode(expr)
            
            full_func = func_template.format(
                name=name, 
                extract_args=extract_args, 
                extract_x_params=extract_x_params, 
                extract_y_params=extract_y_params,
                code=code
            )
            funcs.append(full_func)
    return funcs