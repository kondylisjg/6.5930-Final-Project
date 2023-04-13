## Lab 3: Hardware Modeling and Mapping Exploration
This lab aims to help you get a good understanding of hardware design space exploration and mapping space exploration 
of DNN accelerator designs. Please watch the released [video lectures](http://accelergy.mit.edu/tutorial.html) 
on Accelergy and Timeloop to get a basic understanding of the tools that will be used in this lab.

### First time setup
If you are running this lab for the first time, please open a terminal from Jupyter (`New > Terminal`), and run command:
```
accelergyTables -r /home/workspace/lab3/PIM_estimation_tables
```

### TODO

1. First, we will review modeling of a single processing element (PE) design in Accelergy and Timeloop. Please read questions and instructions in `part01_singlePEArch.ipynb`. This part will cover translating the loop nest into mappings and architectures defined for Timeloop, components for each element in the PE architecture, and basics of how to interpret the results generated by Accelergy and Timeloop.

2. Next, we will expand our discussion into a spatial architecture with multiple PEs. We will review how different configuration of PEs affects the energy and latency, and how emerging technologies can be incorporated to our simulation using Accelergy and Timeloop. Please see `part02_spatialArch.ipynb` for questions and instructions.