__kernel void vector_add(__global const int *a_list, __global const int *b_list, __global int *c_list)
{
    // Get the index of the current index
    int i = get_global_id(0);
    // Calculation
    c_list[i] = a_list[i] + b_list[i];
}