__kernel void avg_processing(
    __global const float *input_arr, 
    __global const float *output_arr, 
    __global const unsigned int *tbl_parameter
    )
{
    // Get the index of the current index
    int i = get_global_id(0);
    // Calculation
    int presRow = i / tbl_parameter[1];
    int presCol = i % tbl_parameter[1];
    if(presRow == 0 || presRow == tbl_parameter[0] - 1 || presCol == 0 || presCol == tbl_parameter[1] - 1)
    { output_arr[i] = input_arr[i]; }
    else 
    {
        output_arr[i] = (input_arr[i - tbl_parameter[1]] + input_arr[i + tbl_parameter[1]] + 
            input_arr[i - 1] + input_arr[i + 1]) / 4
    }
}