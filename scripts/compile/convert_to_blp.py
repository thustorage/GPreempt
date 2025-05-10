import re
func_cnt = 0

def rewrite_cu_file(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()

    # Regex pattern to match __global__ function definitions and declarations
    global_function_pattern = r"(extern \"C\"\s+)?__global__\s+void\s+(\b__launch_bounds__\(\s*\d+\s*(,\s*\d+\s*)?\)\s+)?(\w+)\s*\(([^)]*)\)(\s*\{|\s*;)"
    # Function to rewrite a __global__ function
    def replace_global_function(match):
        global func_cnt
        extern_c = match.group(1) or ""
        launch_bounds = match.group(2) or ""
        function_name = match.group(4)
        params = match.group(5)
        body_or_semicolon = match.group(6).strip()

        # Create the new parameters for the __device__ function
        if params.strip():
            new_params = f"dim3 taskIdx, {params}"
        else:
            new_params = "dim3 taskIdx"

        # Rewrite the __device__ function signature
        device_function_signature = f"__device__ void {function_name}__blp({new_params})"

        if body_or_semicolon == ";":
            # Function declaration
            return f"{device_function_signature};\n{extern_c}__global__ {launch_bounds}void {function_name}({params + ',' if params.strip() else ''} int* preempted, int* pStopIndex, int total_block, int* executed, dim3 shape, int kernel_id);"
        else:
            # Function definition
            # Create the new __global__ function signature
            new_global_params = f"{params}, int* preempted, int* pStopIndex, int total_block, int* executed, dim3 shape, int kernel_id" if params.strip() else "int* preempted, int* pStopIndex, int total_block, int* executed, dim3 shape, int kernel_id"
            global_function_signature = f"{extern_c}__global__ {launch_bounds}void {function_name}({new_global_params})"

            # Construct the body of the new __global__ function
            global_function_body = f"""{{
    while (true) {{
        __shared__ bool stop;
        __shared__ unsigned int x,y,z;
        dim3 taskIdx;

        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{
            stop = *preempted;
            if(stop == 0) {{
                int exec_index = atomicAdd(executed, 1);
                if(exec_index >= total_block) {{
                    stop = 1;
                }}
                x = exec_index % shape.x;
                y = (exec_index / shape.x) % shape.y;
                z = exec_index / (shape.x * shape.y);
            }} else {{
                if(*pStopIndex == -1){{
                    *pStopIndex = kernel_id;        
                }}
            }}
        }}
        __syncthreads();
        if (stop == 1) {{
            return;
        }}
        taskIdx.x = x;
        taskIdx.y = y;
        taskIdx.z = z;
        {function_name}__blp(taskIdx, {', '.join(param.split()[-1].split('*')[-1] for param in params.split(',')) if params.strip() else ''});
    }}
}}"""
            func_cnt += 1
            # Combine the signatures and bodies
            return f"{global_function_signature}\n{global_function_body}\n{device_function_signature}" + "{"

    content = re.sub(r"\bblockIdx\b", "taskIdx", content)
    # Replace __global__ functions
    content = re.sub(global_function_pattern, replace_global_function, content)

    # Replace blockIdx with taskIdx

    # Write the modified content to the output file
    with open(output_file, 'w') as f:
        f.write(content)

import sys
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} raw_code.cu transformed_code.cu")
    exit(0)
    
input_file = sys.argv[1]
output_file = sys.argv[2]
rewrite_cu_file(input_file, output_file)