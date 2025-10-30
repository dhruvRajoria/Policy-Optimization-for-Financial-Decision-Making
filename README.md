# 💰 Policy Optimization for Financial Decision-Making  

**👤 Author:** *Dhruv Rajoria*  
**📊 Dataset:** [LendingClub Loan Data (2007–2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)  
**🎯 Objective:** Build an end-to-end system that uses **Machine Learning** and **Offline Reinforcement Learning (RL)** to optimize loan approval decisions and maximize financial returns.  

---

## 🧠 Project Overview  

This project focuses on building an **intelligent financial decision-making system** for loan approvals.  
It incorporates both **Supervised Learning** and **Offline Reinforcement Learning (RL)** approaches to decide whether to approve or deny a loan, with the goal of **maximizing company profit** and **minimizing risk**.

---

## 🎯 Key Objectives  

1. 🔍 Perform **Exploratory Data Analysis (EDA)** and data preprocessing.  
2. 🤖 Train a **Deep Learning model (MLP)** to predict loan defaults.  
3. ⚙️ Frame the same task as an **Offline RL problem**, using a reward function based on profit/loss.  
4. 📈 Compare both models by **Accuracy**, **F1**, **AUC**, and **Financial Reward** metrics.  

---

# 🚀 How to Run the Code  

### 🧩 Step 1 – Ensure You Have `requirements.txt`  
Make sure the `requirements.txt` file is present in your project directory.  

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 💾 Step 2 – Download the Dataset  
Download the **LendingClub Loan Data (2007–2018)** dataset from Kaggle:  
🔗 [https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Once downloaded, locate the following file:  
```
accepted_2007_to_2018Q4.csv.gz
```

---

### 📁 Step 3 – Download the Project Files  
Ensure that you have the following files in your project folder:  

```
EDA_and_Preprocessing.ipynb
DeepLearning_model.ipynb
RL_baseline_agent.ipynb
Analysis_and_Comparison.ipynb
main.py
requirements.txt
```

---

### 🛠️ Step 4 – Update the Dataset Path  
In the first cell of the notebook **`EDA_and_Preprocessing.ipynb`**, update the dataset path to where you have saved it locally.  

```python
RAW_GZ = "path/to/accepted_2007_to_2018Q4.csv.gz"
```

---

### ▶️ Step 5 – Run the Main Script  

Run the main script to execute the entire project pipeline automatically.  

```bash
python main.py
```

This script will:  
- 🧹 Perform **EDA and preprocessing**  
- 🧠 Train **Deep Learning and RL models**  
- ⚙️ Run all four notebooks sequentially using `papermill`  
- 💾 Save all results in the following folders:  
  ```
  data/
  models/
  analysis/
  ```

---

### 📊 Step 6 – View Results and Reports  

After successful execution:  
- 📁 Check the generated outputs and metrics in the **`analysis/`** folder.  
- 📘 Review the visualizations and model comparisons in the executed notebooks.  

---

## 🧩 Tools & Technologies  

| Category | Tools / Libraries |
|-----------|------------------|
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, TensorFlow / PyTorch |
| **Reinforcement Learning** | Stable-Baselines3 / Custom RL |
| **Automation** | Papermill, Jupyter |
| **Environment** | Python 3.9+ |

---

## 📘 Summary  

- The **MLP model** predicts loan defaults effectively using supervised learning.  
- The **Offline RL agent** focuses on decision-making that maximizes profit rather than prediction accuracy.  
- Comparing both models highlights the trade-off between **predictive performance** and **financial gain**.  
