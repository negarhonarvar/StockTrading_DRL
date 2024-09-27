# StockTrading_DRL
Automated stock trading strategy using deep reinforcement learning and recurrent neural networks.
## Introduction:
We begin by discussing the challenges in stock trading, particularly the issues related to noisy and irregular data. The proposed solution is a Deep Reinforcement Learning (DRL) model with an architecture designed to handle these challenges by learning from historical stock data.
## Dataset:
The dataset includes stock prices from major companies like NVIDIA, Microsoft, Apple, Amazon, and Google, covering the period from January 1, 2009, to June 1, 2024. Due to significant market events like the 2020 pandemic and the start of wars, the dataset is divided into training (2009-2022) and testing (2022-2024) periods. A metric called "Turbulence Threshold" is introduced to handle extreme market fluctuations.

<img src="https://github.com/user-attachments/assets/171e2bca-7481-413b-b0c9-b18e35907f5e" width="400">

## Data Processing:
Before training the model, the data undergoes several preprocessing steps, such as normalization and feature extraction. The importance of data consistency, the impact of weighting data points, and strategies for maintaining data integrity are emphasized.
## Environment Definition:
The environment is designed using the OpenAI framework, simulating a stock market where the agent (trading algorithm) can buy, sell, or hold stocks. The action space is defined as discrete actions corresponding to buying, selling, or holding stocks, while the observation space includes various stock-related metrics like prices, indicators, etc.
## Model Architecture:
The proposed model is a CLSTM-PPO (Cascading Long Short-Term Memory - Proximal Policy Optimization) model. It uses LSTM layers to capture temporal dependencies in stock data and a PPO algorithm to optimize trading decisions. The model is trained to maximize cumulative returns while minimizing risks like maximum pullback.

<img src="https://github.com/user-attachments/assets/3779b6d7-b0d8-4762-819d-d88ca07030b7" width="400">

<img src="https://github.com/user-attachments/assets/ea280a11-41d6-415a-a631-cc9d20d8257c" width="400">

<img src="https://github.com/user-attachments/assets/ef5cc95a-cafa-4ca5-9b0f-128421a0e9e7" width="400">

The following algorithm summarizes the process of our work:

<img src="https://github.com/user-attachments/assets/9c327872-132a-4f25-beb1-9daecd32c0ea" width="400">


## Evaluation Metrics:
The model’s performance is evaluated using several financial metrics:

    Cumulative Return (CR)
    Max Earning Rate (MER)
    Maximum Pullback (MPB)
    Average Profitability Per Trade (APPT)
    Sharpe Ratio (SR)

<img src="https://github.com/user-attachments/assets/c4e6551c-d75c-409d-b2d8-5af1c2c51e76" width="400">

<img src="https://github.com/user-attachments/assets/69e0ec3b-3245-447c-a3fe-c004d437cf57" width="400">

<img src="https://github.com/user-attachments/assets/a93cb337-4c42-4e24-92a6-8d0a466b20d6" width="400">

<img src="https://github.com/user-attachments/assets/c5f1ab46-cde9-4792-9c2a-1f2976074a75" width="400">

<img src="https://github.com/user-attachments/assets/ebc7cd26-e919-43e6-8b42-639fd495e934" width="400">


These metrics help assess the profitability, risk, and overall performance of the trading strategy.
## Hyperparameters:
1. Time Window Size:

    This hyperparameter defines the length of the sequence of data points (e.g., days of stock prices) that the model considers as input. It is crucial for capturing patterns over time.
    Impact:
        A larger time window allows the model to capture long-term trends and dependencies in the data. However, this also increases the complexity of the model and the computational resources required.
        A smaller time window focuses on short-term patterns, which might miss out on broader trends but can react faster to recent changes.
    Values Tested: The report mentions testing different time window sizes, such as 5, 15, 30, and 50. The optimal window size depends on the specific dataset and trading strategy, with the report finding that larger windows generally improve performance but come with trade-offs.

2. Hidden Size of LSTM Networks:

    This refers to the number of units in the hidden layers of the LSTM (Long Short-Term Memory) networks, which are used to model the temporal dependencies in the stock data.
    Impact:
        Larger hidden sizes allow the model to capture more complex patterns and interactions in the data, which can improve accuracy but also increase the risk of overfitting, especially with limited data.
        Smaller hidden sizes reduce the risk of overfitting and computational cost but might miss out on capturing intricate relationships in the data.
    Considerations: The report suggests that the hidden size should be carefully tuned based on the complexity of the data and the amount of available training data.

3. Number of Time Steps:

    This hyperparameter defines the number of steps or sequence lengths that the model processes at once during training.
    Impact:
        More time steps allow the model to consider a longer sequence of past events, which can be useful for understanding long-term dependencies. However, this can also lead to increased computational costs and potential overfitting.
        Fewer time steps make the model focus more on immediate past events, which might speed up training but could miss out on important historical information.
    Value Used: The report mentions a large number of time steps (e.g., 10,000) to allow the model to learn from extensive historical data.

4. Boolean Parameter for State Termination:

    This hyperparameter controls when the model should terminate a sequence or trajectory during training. It is a Boolean parameter that decides whether to end the current state based on certain conditions.
    Impact:
        If set to True: The model will terminate the sequence early, which can prevent overfitting by not allowing the model to focus too long on any particular pattern.
        If set to False: The model will continue learning from the current sequence, which might be useful for capturing long-term dependencies but could lead to overfitting.
    Use Case: This parameter allows for flexibility in training, making the model adaptable to different market conditions by controlling how long it should focus on specific sequences.

5. Architectural Hyperparameters:

    These are additional hyperparameters related to the overall training process, including:
        Learning Rate: Determines how quickly the model adjusts its parameters in response to the gradients.
        Batch Size: Defines the number of samples processed before the model's parameters are updated.
        Number of Epochs: Indicates how many times the entire dataset is passed through the model during training.
    Impact:
        Learning Rate: A higher learning rate can speed up training but may overshoot the optimal solution, while a lower rate ensures more precise updates but can slow down the process.
        Batch Size: Larger batches make the gradient estimation more stable but require more memory, while smaller batches allow for more frequent updates but with noisier gradients.
        Number of Epochs: Too few epochs might result in underfitting, while too many can lead to overfitting.
    Values Used: The report mentions using a batch size of 64 and 128, and the number of epochs being tested around 10 and 15, indicating an attempt to balance training time and model performance.

6. Stable Baseline Parameters:

   These parameters are related to the specific implementation of the Proximal Policy Optimization (PPO) algorithm within the Stable Baselines library. They include:
        Value Function Coefficient: Weights the contribution of the value function in the loss calculation.
        Advantage Estimation: Adjusts how the advantages (differences between expected and actual rewards) are calculated and used in training.
    Implementation: The report suggests setting these parameters according to the recommendations in the related literature or default settings within the Stable Baselines framework to ensure the PPO algorithm performs optimally.

   <img src="https://github.com/user-attachments/assets/0eaa416d-accd-4537-a1b1-9ddde14f7723" width="400">

## Results:
It Is necessary to highlight how larger time windows generally leads to better model performance, but with diminishing returns and increased computational costs.

<img src="https://github.com/user-attachments/assets/6ebc4c26-94c3-47ac-b880-69bf16f5d6c0" width="400">

<img src="https://github.com/user-attachments/assets/a46ad482-5b82-42d7-a327-da11200adbf0" width="400">

<img src="https://github.com/user-attachments/assets/e1d6d4d6-a0c7-4142-96da-ee62baec7149" width="400">

<img src="https://github.com/user-attachments/assets/dd8c5a06-3eaf-4060-8ff6-f2f00be23464" width="400">

<img src="https://github.com/user-attachments/assets/4d04fe87-3073-4279-9315-09d97d8bc085" width="400">


## Conclusion:
While the proposed method can yield profitable trading strategies, it is also sensitive to market conditions and requires careful tuning of hyperparameters.
