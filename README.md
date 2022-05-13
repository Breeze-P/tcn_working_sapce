# Vehicle following trajectory optimization model based on TCN

## Overview

This is a Vehicle following trajectory optimization model based on Temporal Convolutional Network, produced by LV Yanming, ZHAO Zhiyu and GUI Shiyang.

## File Structure

- The file consists of 6 part: source code, model, graphs, comparison, generalization and dataProcessing.
- In source code folder, you can find the source code of our model.
- All trained network models are saved in the model folder.
- Graphs regarding the project are in the graphs folder, including the Scenario Simulation, TCN's structure, Parameter Explanation about the model, loss curve of other network models and Sequence Modeling Test about the car following problem. Meanwhile, the generalization of each network model is also provided.
- In comparison folder, you can see all the network models used for comparison, including LSTM, GRU and RNN.
- Source codes in generalization folder can convert data in HighD dataset to data used for generalization.
- All codes used for data processing is placed in the dataProcessing folder, where you can convert datasets to data suitable for the project.

## Environment

- Python version: 3.7
- Pytorch version: 1.10.1(used in TCN model)
- Tensorflow version: 2.7.0 (used in LSTM, GRU and RNN model)
- matplotlib version: 3.4.2
- pandas version: 1.2.4

## Parameters

![](https://github.com/Breeze-P/tcn_working_sapce/blob/lym/car_following/graphs/Parameter%20Explanation.png#pic_center =100%x100%)

