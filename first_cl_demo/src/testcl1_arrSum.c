#pragma warning (disable: 4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE 0x1000

int main()
{
    // Create two vectors
    int i;
    const int list_size = 1024;
    int *a_list = (int*)malloc(sizeof(int) * list_size);
    int *b_list = (int*)malloc(sizeof(int) * list_size);
    for (i = 0; i < list_size; i++)
    {
        a_list[i] = i;
        b_list[i] = list_size - i;
    }

    // Load the kernel codes into source_str
    FILE *fp_kernel = NULL;
    char *source_str;
    size_t source_size;

    fp_kernel = fopen("src/vector_add_kernel.cl", "r");
    if(!fp_kernel)
    {
        fprintf(stderr, "Failed to load kernel!\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp_kernel);
    fclose(fp_kernel);

    // Obtain the platform and devices information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platform;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platform);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create context for OpenCL
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    // Create command queue 
    // Here do not use the old version "clCreateCommandQueue" which has been deprecated before CL 2.0
    // Use "clCreateCommandQueueWithProperties" instead!
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context,device_id, 0, &ret);
    // Create buffer for each vector on the device
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, list_size * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, list_size * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, list_size * sizeof(int), NULL, &ret);

    // Copy list_a and list_b into the GPU memory
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, list_size * sizeof(int), a_list, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, list_size * sizeof(int), b_list, 0, NULL, NULL);

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    // Construct program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    // Create OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set kernel parameters
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);

    // Execute the kernel
    size_t global_item_size = list_size;    // Handle the whole list
    size_t local_item_size = 64;            // Cut into 64 pieces
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Read ram buffer C into the local buffer C
    int *c_list = (int*)malloc(sizeof(int) * list_size);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, list_size * sizeof(int), c_list, 0, NULL, NULL);

    // Write result into a file
    FILE *fp_result = NULL;
    char szBuf[128] = {0};

    fp_result = fopen("dep/result.txt", "w");
    if(!fp_result)
    {
        fprintf(stderr, "Unable to create an output file!\n");
        exit(2);
    }
    for(i = 0; i < list_size; i++)
    {
        memset(szBuf, 0, 128);
        sprintf(szBuf, "%d + %d = %d\n", a_list[i], b_list[i], c_list[i]);
        fwrite(szBuf, strlen(szBuf), 1, fp_result);
    }
    fclose(fp_result);

    // Clear the resource
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(a_list);
    free(b_list);
    free(c_list);
    return 0;
}