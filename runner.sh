#!/bin/bash

echo "🧪 Running Random Forest..."
python dl_rf.py
python eval_plt.py model_rf.pkl

echo "🧪 Running XGBoost..."
python dl_xgb.py
python eval_plt.py model_xgb.pkl

echo "🧪 Running Neural Net..."
python dl_nn.py
python eval_plt.py model_nn.pkl

echo "✅ All models evaluated. Check the PNG plots for results!"

