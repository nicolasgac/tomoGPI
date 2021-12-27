__kernel void hello_world(int thread_id_from_which_to_print_message){
  // Get index of the work item
  unsigned thread_id = get_global_id(0);

  if(thread_id == thread_id_from_which_to_print_message) {
    printf("Thread #%u: Hello from OpenCL CPU or GPU!\n", thread_id);
  }
}
