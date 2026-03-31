# 🎯 Customer Churn Prediction System

Predict customer churn risk using machine learning. An AI-powered web app that identifies at-risk customers in seconds.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser → http://localhost:8501
```

That's it! You're ready to make predictions.

## 🌐 Live Demo

**Try it online:** https://customer-churn-predict-system.streamlit.app/

No installation needed! Start making predictions instantly.

## ✨ What This Does

- **Predict Churn Risk**: Get instant churn probability (0-100%)
- **Risk Levels**: Low 🟢 | Medium 🟡 | High 🔴
- **Interactive Dashboard**: Charts, metrics, and customer profiles
- **Real-time Results**: Analyze customer data in seconds

## � Tech Stack

- **Streamlit** - Web interface
- **TensorFlow/Keras** - Neural network model
- **Scikit-learn** - Data preprocessing
- **Plotly** - Interactive charts
- **Python 3.8+**

## 📁 Project Files

```
├── app.py                    # Main app - run this!
├── churn_model.h5           # Trained AI model
├── Churn_Modelling.csv      # Training data
├── Experiments.ipynb        # Model training notebook
├── predictions.ipynb        # Example predictions
├── requirements.txt         # Python packages to install
├── scaler.pkl              # Data normalizer
├── label_encoder.pkl       # Gender encoder
├── onehot_encoder.pkl      # Location encoder
└── logs/                   # Training logs
```

## � Installation

**Requirements:** Python 3.8+

```bash
# 1. Open terminal/command prompt
cd "e:\ANN Churn Project"

# 2. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 3. Install packages
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

## � How to Use

1. **Fill the Form** (left sidebar):
   - Enter customer details (location, age, salary, etc.)
   
2. **Click "Predict"** 
   - Get churn probability instantly
   
3. **Review Results**:
   - Churn probability % (0-100)
   - Risk level (Low/Medium/High)
   - Charts and customer profile
   - Retention probability

**That's it!** No complex steps needed.

## 🧠 The AI Model

**What it does:** Analyzes 10 customer features to predict if they'll leave (churn).

**Input Data:**
- Location (Geography)
- Gender, Age
- Credit Score, Salary, Balance
- Products used, Credit Card, Activity status
- Years as customer (Tenure)

**Output:**
- Churn probability (0-100%)
- High probability = likely to leave
- Low probability = likely to stay

## 📊 Dataset

- **File:** `Churn_Modelling.csv`
- **Records:** 10,000 customer records
- **Used for:** Training the neural network model

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" error | Run `pip install -r requirements.txt` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Model not found error | Make sure `churn_model.h5` is in the project folder |

## 📚 Additional Resources

- **Want to see training details?** Open `Experiments.ipynb`
- **Want example predictions?** Open `predictions.ipynb`
- **View training logs:** Run `tensorboard --logdir=logs/fit`

## 📄 License

MIT License - Feel free to use this project freely.

---

**Status:** ✅ Ready to use  
**Last Updated:** March 2026
