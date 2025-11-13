import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
import os
from glob import glob
import matplotlib.pyplot as plt 
from itertools import product 

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# ----- 1. File Loading Functions (Cached for speed) -----

@st.cache_data
def load_model(pkl_file="gam_model.pkl"):
    """Loads the pickled GAM model."""
    try:
        with open(pkl_file, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {pkl_file}")
        return None

@st.cache_data
def load_data_info():
    """Loads feature lists, bounds, and default values from script artifacts."""
    try:
        with open("gam_summary.json", "r") as f:
            features = json.load(f)["features"]
        
        df_raw = pd.read_csv("Data_raw_CSV.csv")
        bounds = {}
        for col in features:
            bounds[col] = (float(df_raw[col].min()), float(df_raw[col].max()))
            
        df_best = pd.read_csv("gam_best_params.csv")
        defaults = df_best[features].iloc[0].to_dict()
        
        return features, bounds, defaults
        
    except FileNotFoundError as e:
        st.error(f"A required data file was not found: {e.filename}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# ----- Optimization Function (No changes) -----
@st.cache_data
def find_optimal_T_RT_HR(_model, fixed_inputs_tuple, all_features, bounds_dict, grid_points=15):
    """
    Performs a grid search for Temp, RT, and HR given all other fixed inputs.
    We use a tuple of fixed inputs as the cache key.
    """
    target_vars = ["Temp", "RT", "HR"]
    base_inputs = dict(fixed_inputs_tuple) 

    if not all(v in all_features for v in target_vars):
        st.error("Model features list does not include 'Temp', 'RT', and 'HR'.")
        return None

    temp_grid = np.linspace(bounds_dict["Temp"][0], bounds_dict["Temp"][1], grid_points)
    rt_grid = np.linspace(bounds_dict["RT"][0], bounds_dict["RT"][1], grid_points)
    hr_grid = np.linspace(bounds_dict["HR"][0], bounds_dict["HR"][1], grid_points)
    
    grid = list(product(temp_grid, rt_grid, hr_grid))
    
    test_df = pd.DataFrame([base_inputs] * len(grid))
    
    grid_df = pd.DataFrame(grid, columns=target_vars)
    test_df["Temp"] = grid_df["Temp"]
    test_df["RT"] = grid_df["RT"]
    test_df["HR"] = grid_df["HR"]
    
    test_df = test_df[all_features]
    
    predictions = _model.predict(test_df.values)
    
    best_index = np.argmax(predictions)
    best_yield = predictions[best_index]
    best_params = test_df.iloc[best_index]
    
    return {
        "Temp": best_params["Temp"],
        "RT": best_params["RT"],
        "HR": best_params["HR"],
        "Best_Yield": best_yield
    }

# ----- 2. Load Model and Data -----
model = load_model()
features, bounds, defaults = load_data_info()

if model is None or features is None:
    st.stop()

# ----- 3. Main Interface -----
st.title("🧪 GAM Model Interactive Dashboard")
st.write("Use the sidebar to adjust input parameters and see the predicted Yield-char.")

# ----- 4. Sidebar for Inputs -----
st.sidebar.header("🛠️ Input Parameters")
user_inputs = {}
for feat in features:
    min_val, max_val = bounds[feat]
    default_val = defaults[feat]
    
    # --- NEW: (Request 1) Add bounds to the label ---
    label_with_bounds = f"{feat} (Range: {min_val:.2f} to {max_val:.2f})"
    # --- End NEW ---

    user_inputs[feat] = st.sidebar.number_input(
        label=label_with_bounds, # Use the new label
        min_value=min_val, 
        max_value=max_val,
        value=default_val, 
        format="%.3f"
    )

# ----- 5. Prediction Calculation and Display -----
st.header("📈 Prediction")

input_data = [user_inputs[feat] for feat in features]
X_new = np.array([input_data])

try:
    prediction = model.predict(X_new)
    st.metric(
        label="Predicted Yield-char (%) (at current settings)",
        value=f"{prediction[0]:.3f}"
    )
except Exception as e:
    st.error(f"Could not make prediction: {e}")

# ----- Optimization Section (No changes) -----
st.subheader("🤖 Optimize Key Parameters")
st.write("Click this button to find the best `Temp`, `RT`, and `HR` based on all *other* parameters you've set in the sidebar.")

if st.button("Suggest Best Temp, RT, and HR"):
    fixed_inputs_dict = user_inputs.copy()
    for var in ["Temp", "RT", "HR"]:
        fixed_inputs_dict.pop(var, None) 
    
    fixed_inputs_tuple = tuple(sorted(fixed_inputs_dict.items()))
    
    with st.spinner("Searching 3,375 combinations..."):
        result = find_optimal_T_RT_HR(model, fixed_inputs_tuple, features, bounds)
    
    if result:
        st.success(f"Optimized Yield: {result['Best_Yield']:.3f}%")
        st.info("With your current settings, use these values to maximize yield:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Temp", f"{result['Temp']:.2f}")
        col2.metric("Best RT", f"{result['RT']:.2f}")
        col3.metric("Best HR", f"{result['HR']:.2f}")

st.divider()

# ----- 6. Display Model Analysis Artifacts -----
st.header("📊 Model Analysis Results")

tab1, tab2, tab3 = st.tabs(["Model Performance", "Interactive 1D Plots", "Interactions (3D)"])

with tab1:
    st.subheader("Overall Model Performance")
    if os.path.exists("cv_parity.png"):
        st.image("cv_parity.png", caption="Actual vs. Predicted (OOF)")
    st.subheader("Top 10 Feature Importance")
    if os.path.exists("importance_permutation_top10.png"):
        st.image("importance_permutation_top10.png", caption="Permutation Importance")

# --- NEW: (Request 2) Modify Tab 2 to show all plots ---
with tab2:
    st.subheader("Interactive 1D Plots (Ceteris Paribus)")
    st.write("These plots show the real-time effect of *one* feature, holding all others constant at their sidebar values.")
    
    # Add a performance warning
    st.info("ℹ️ **Note:** All graphs on this page are recalculated every time you change a value in the sidebar. This may be slow if you have many features.")

    # Use columns to make the layout more compact
    col1, col2 = st.columns(2)
    
    # Loop through ALL features instead of using a selectbox
    for i, selected_feat in enumerate(features):
        
        min_val, max_val = bounds[selected_feat]
        X_grid = np.linspace(min_val, max_val, 100) # 100 points per plot

        # Create scenario data for this specific feature
        scenario_data = pd.DataFrame([user_inputs] * 100)
        scenario_data[selected_feat] = X_grid
        scenario_data_ordered = scenario_data[features] 

        try:
            # Predict for all 100 points
            y_pred = model.predict(scenario_data_ordered.values)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(6, 4)) # Set a consistent size
            ax.plot(X_grid, y_pred, color="blue", linewidth=2)
            
            # Show the current selected point
            current_val = user_inputs[selected_feat]
            current_pred = prediction[0] 
            ax.plot(current_val, current_pred, 'ro', markersize=8, label="Current Selection")
            
            # Show the bounds
            ax.axvline(min_val, color='gray', linestyle='--')
            ax.axvline(max_val, color='gray', linestyle='--')
            
            ax.set_xlabel(selected_feat)
            ax.set_ylabel("Predicted Yield (%)")
            ax.set_title(f"Real-time effect of {selected_feat}")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize='small')
            
            # Place plot in the correct column
            if i % 2 == 0:
                col1.pyplot(fig)
            else:
                col2.pyplot(fig)

        except Exception as e:
            if i % 2 == 0:
                col1.error(f"Error plotting {selected_feat}: {e}")
            else:
                col2.error(f"Error plotting {selected_feat}: {e}")

# --- End NEW ---

with tab3:
    st.subheader("3D Interaction Surfaces (Static)")
    st.write("These 3D plots show the combined effect of two features (they do not update in real-time).")
    
    surface_files = sorted(glob("surface3d_*.png")) 
    if surface_files:
        for img_file in surface_files:
            st.image(img_file)
    else:
        st.info("No 3D Surface files (surface3d_*.png) found.")