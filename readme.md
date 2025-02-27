# ⏳ Pendulum-v1 Reinforcement Learning

Welcome to the Pendulum-v1 Reinforcement Learning Project! 🚀 This repository implements Q-learning to train an agent to balance a pendulum using discrete action space.

![pendulum](https://github.com/user-attachments/assets/39dc248a-be6a-45fe-b679-8a37fef3b39b)


## 🎯 Project Overview

The Pendulum-v1 environment is a classic reinforcement learning problem where the goal is to keep a pendulum upright by applying torques. The project discretizes the continuous state and action space and applies Q-learning to optimize the control strategy.

## 🏗 Features

✅ Discretized State and Action Space – Converts continuous observations into discrete bins for Q-learning.\
✅ Epsilon-Greedy Exploration – Implements exploration-exploitation trade-off.\
✅ Reward Shaping – Encourages better balancing by penalizing deviation.\
✅ Video Recording – Saves the best episode as a video.\
✅ Pretrained Model Saving – Stores and loads the best Q-table.

## 📂 Repository Structure
```
Pendulum/
│── main.py                  # Training and evaluation script
│── pendulum_q_table.npy      # Saved Q-table
│── Videos/ pendulum.png      # Stores successful episode
│── requirements.txt         # Dependencies
│── README.md                # Documentation
```

## 🛠 Setup and Installation

1️⃣ Clone the Repository
```
  git clone https://github.com/Ajith-Kumar-Nelliparthi/Pendulum.git
  cd Pendulum
```

## 2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed.
```
  pip install -r requirements.txt
```
## 🚀 Training the Agent

Run the following command to start training the agent:
```
  python main.py
```

The script will train the agent for 20,000 episodes and store the best Q-table in pendulum_q_table.npy.

## 🎥 Recording the Best Episode

The best-performing episode is saved as a video in the Videos/ directory after training.

## 🏆 Testing the Trained Model

Once training is complete, you can test the model using:
```
  python main.py --test
```
This will load the saved Q-table and run the agent in a human-rendered environment.

## 📊 Performance Metrics

Best Score Tracking – Keeps track of the highest reward achieved.

Epsilon Decay – Improves learning by reducing exploration over time.

## 🖼️ Simple Physical Pendulum

## 📜 References

[OpenAI Gym](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

[Q-learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)

## 🤝 Contributing

Feel free to fork this repository and contribute with improvements!

## 📧 Contact

For any questions, reach out to: \

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/Ajith532542840)\
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nelliparthi-ajith-233803262)\
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](nelliparthi123@gmail.com)

## 🌟 If you like this project, give it a star! ⭐

















