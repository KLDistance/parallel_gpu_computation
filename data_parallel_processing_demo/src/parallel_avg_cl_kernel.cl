__kernel void avg_processing(__global float *org_data_list, __global unsigned int *arr_info, __global float *new_data_list)
{
    int i = get_global_id(0);
    // Calculation
    int presRow = i / arr_info[1];
    int presCol = i % arr_info[1];
    if(presRow == 0 || presRow == arr_info[0] - 1 || presCol == 0 || presCol == arr_info[1] - 1)
    {
        new_data_list[i] = org_data_list[i];
    }
    else
    {
        new_data_list[i] = (org_data_list[i - arr_info[1]] + org_data_list[i + arr_info[1]] + 
            org_data_list[i - 1] + org_data_list[i + 1]) / 4;
    }

}