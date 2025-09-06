import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.set_page_config(page_title="Comparison of input and output via gaussian curve fitting", layout="wide")
st.title("Comparison of input and output via gaussian curve fitting")

st.markdown("""
Paste or upload **two-column datasets** (Distance, Intensity) for comparison.
The app fits each dataset as a sum of Gaussians and allows computing the effective seperation coefficient between two selected peaks.
Each panel has **independent baseline correction** options.
""")

# -----------------------------
# Helper functions
# -----------------------------
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

def multi_gaussian(x, *params):
    y_sum = np.zeros_like(x, dtype=float)
    for i in range(0, len(params), 3):
        amp, cen, wid = params[i:i+3]
        y_sum += gaussian(x, amp, cen, wid)
    return y_sum

def fit_gaussian(x, y, n_peaks):
    try:
        amps_guess = [max(y)/n_peaks]*n_peaks
        cens_guess = np.linspace(min(x), max(x), n_peaks)
        wids_guess = [(max(x)-min(x))/(4*n_peaks)]*n_peaks
        p0 = []
        for a,c,w in zip(amps_guess, cens_guess, wids_guess):
            p0.extend([a,c,w])
        popt, _ = curve_fit(multi_gaussian, x, y, p0=p0, maxfev=10000)
        y_fit = multi_gaussian(x, *popt)
        return popt, y_fit
    except Exception as e:
        st.error(f"Gaussian fitting failed: {str(e)}")
        # Return dummy values to prevent crashes
        dummy_popt = [max(y)/n_peaks, np.mean(x), (max(x)-min(x))/4] * n_peaks
        dummy_y_fit = np.zeros_like(x)
        return dummy_popt, dummy_y_fit

# -----------------------------
# Input panels side by side
# -----------------------------
col1, col2 = st.columns(2)
datasets = []

for idx, col in enumerate([col1, col2], start=1):
    with col:
        st.subheader(f"Dataset {idx}")
        method = st.radio(f"Input Method {idx}", ["Paste Data", "Upload CSV"], key=f"method{idx}")
        df = None
        if method=="Paste Data":
            data_str = st.text_area(f"Paste data for Dataset {idx}", key=f"paste{idx}")
            if data_str:
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(data_str), sep=r'\s+', engine="python")
                    # Check if we have at least 2 columns
                    if len(df.columns) < 2:
                        st.error("Data must have at least 2 columns (Distance, Intensity)")
                        df = None
                except Exception as e:
                    st.error(f"Failed to parse pasted data: {str(e)}")
        else:
            file = st.file_uploader(f"Upload CSV for Dataset {idx}", type=["csv"], key=f"file{idx}")
            if file:
                try:
                    df = pd.read_csv(file)
                    # Check if we have at least 2 columns
                    if len(df.columns) < 2:
                        st.error("CSV must have at least 2 columns (Distance, Intensity)")
                        df = None
                except Exception as e:
                    st.error(f"Failed to read CSV file: {str(e)}")
        
        if df is not None:
            st.dataframe(df.head())
            
            try:
                x = np.array(df.iloc[:,0], dtype=float)
                y = np.array(df.iloc[:,1], dtype=float)
                
                # Check for valid data
                if len(x) < 3:
                    st.error("Need at least 3 data points for fitting")
                    datasets.append(None)
                    continue
                    
                if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                    st.warning("Data contains NaN values, removing them...")
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x = x[mask]
                    y = y[mask]
                    
                if len(x) < 3:
                    st.error("Not enough valid data points after removing NaN values")
                    datasets.append(None)
                    continue

                # -----------------------------
                # Baseline correction
                # -----------------------------
                st.markdown("**Baseline Correction**")
                baseline_type = st.selectbox(f"Baseline Type Dataset {idx}", ["None", "Constant", "Linear", "Polynomial"], key=f"baseline_type{idx}")
                
                if baseline_type == "Constant":
                    const_base = st.number_input(f"Constant baseline value Dataset {idx}", value=0.0, step=0.1, key=f"const{idx}")
                    y = y - const_base
                elif baseline_type == "Linear":
                    m = st.number_input(f"Slope (m) Dataset {idx}", value=0.0, step=0.01, key=f"slope{idx}")
                    b = st.number_input(f"Intercept (b) Dataset {idx}", value=0.0, step=0.1, key=f"intercept{idx}")
                    y = y - (m*x + b)
                elif baseline_type == "Polynomial":
                    poly_order = st.selectbox(f"Polynomial order Dataset {idx}", [1, 2, 3, 4, 5], index=1, key=f"poly_order{idx}")
                    
                    # Option to fit polynomial automatically or enter coefficients manually
                    poly_method = st.radio(f"Polynomial method Dataset {idx}", ["Auto-fit", "Manual coefficients"], key=f"poly_method{idx}")
                    
                    if poly_method == "Auto-fit":
                        # Option to select baseline points
                        use_endpoints = st.checkbox(f"Use only endpoint regions Dataset {idx}", value=True, key=f"endpoints{idx}")
                        
                        if use_endpoints:
                            endpoint_percent = st.slider(f"Endpoint percentage Dataset {idx}", 5, 25, 10, key=f"end_pct{idx}")
                            n_points = len(x)
                            n_end = int(n_points * endpoint_percent / 100)
                            
                            # Use first and last portions for baseline fitting
                            x_baseline = np.concatenate([x[:n_end], x[-n_end:]])
                            y_baseline = np.concatenate([y[:n_end], y[-n_end:]])
                        else:
                            x_baseline = x
                            y_baseline = y
                        
                        try:
                            # Fit polynomial to baseline points
                            poly_coeffs = np.polyfit(x_baseline, y_baseline, poly_order)
                            baseline = np.polyval(poly_coeffs, x)
                            
                            # Show the fitted polynomial equation
                            poly_eq = " + ".join([f"{coeff:.3e}×x^{poly_order-i}" if i != poly_order else f"{coeff:.3e}" 
                                                 for i, coeff in enumerate(poly_coeffs)])
                            st.write(f"**Fitted baseline:** {poly_eq}")
                            
                        except Exception as e:
                            st.error(f"Polynomial fitting failed: {str(e)}")
                            baseline = np.zeros_like(x)
                    
                    else:  # Manual coefficients
                        st.write(f"**Enter coefficients for polynomial of order {poly_order}:**")
                        poly_coeffs = []
                        for i in range(poly_order + 1):
                            power = poly_order - i
                            coeff_key = f"coeff_{power}_{idx}"
                            if power == 0:
                                coeff = st.number_input(f"Constant term", value=0.0, step=0.1, key=coeff_key)
                            elif power == 1:
                                coeff = st.number_input(f"x coefficient", value=0.0, step=0.01, key=coeff_key)
                            else:
                                coeff = st.number_input(f"x^{power} coefficient", value=0.0, step=0.001, format="%.6f", key=coeff_key)
                            poly_coeffs.append(coeff)
                        
                        baseline = np.polyval(poly_coeffs, x)
                    
                    # Subtract baseline
                    y_original = y.copy()  # Keep original for plotting
                    y = y - baseline
                    
                    # Show baseline correction plot
                    if st.checkbox(f"Show baseline correction Dataset {idx}", key=f"show_baseline{idx}"):
                        fig_baseline, ax_baseline = plt.subplots(figsize=(6,3))
                        ax_baseline.plot(x, y_original, 'b-', label="Original", alpha=0.7)
                        ax_baseline.plot(x, baseline, 'g--', label="Baseline", linewidth=2)
                        ax_baseline.plot(x, y, 'r-', label="Corrected")
                        ax_baseline.set_xlabel("Distance")
                        ax_baseline.set_ylabel("Intensity")
                        ax_baseline.legend()
                        ax_baseline.set_title(f"Baseline Correction - Dataset {idx}")
                        st.pyplot(fig_baseline, clear_figure=True)
                
                # Optional Image Upload
                # -----------------------------
                st.markdown("**Reference Photo (optional)**")
                img_file = st.file_uploader(f"Upload image for Dataset {idx}", type=["png", "jpg", "jpeg"], key=f"img{idx}")
                if img_file:
                    st.image(img_file, caption=f"Dataset {idx} reference image", use_column_width=True)            
                
                # -----------------------------
                # Gaussian fitting
                # -----------------------------
                n_peaks = st.number_input(f"Number of Gaussian Peaks Dataset {idx}", min_value=1, max_value=10, value=2, key=f"n{idx}")
                
                # Check if we have enough data points for the number of peaks
                min_points_needed = n_peaks * 3  # 3 parameters per peak
                if len(x) < min_points_needed:
                    st.warning(f"Warning: Only {len(x)} data points available, but {min_points_needed} needed for {n_peaks} peaks. Consider reducing number of peaks.")
                
                popt, y_fit = fit_gaussian(x, y, n_peaks)
                datasets.append((x, y, popt, n_peaks))
                
                # Plot
                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(x, y, 'b.', label="Data")
                ax.plot(x, y_fit, 'r-', label="Fit")
                for i in range(n_peaks):
                    amp, cen, wid = popt[i*3:(i+1)*3]
                    # Check for reasonable width values to avoid plotting issues
                    if wid > 0:
                        ax.plot(x, gaussian(x, amp, cen, wid), '--', label=f"Peak {i+1}")
                ax.set_xlabel("Distance")
                ax.set_ylabel("Intensity")
                ax.legend()
                st.pyplot(fig, clear_figure=True)
                
                # Fitted table
                peak_table = pd.DataFrame({
                    "Peak": [f"Peak {i+1}" for i in range(n_peaks)],
                    "Amplitude": [f"{popt[i*3]:.3f}" for i in range(n_peaks)],
                    "Center": [f"{popt[i*3+1]:.3f}" for i in range(n_peaks)],
                    "Width": [f"{popt[i*3+2]:.3f}" for i in range(n_peaks)],
                    "FWHM": [f"{2.3548*popt[i*3+2]:.3f}" for i in range(n_peaks)]
                })
                st.dataframe(peak_table)
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                datasets.append(None)
        else:
            datasets.append(None)

# -----------------------------
# Effective Seperation Coefficient Cf section
# -----------------------------
st.subheader("Compute Effective Seperation Coefficient Cf between Two Peaks")
if len(datasets)==2 and all(datasets):
    dataset_idx = st.selectbox("Select dataset for Cf calculation", [1,2])
    x, y, popt, n_peaks = datasets[dataset_idx-1]
    
    if n_peaks >= 2:
        peak_indices = list(range(1, n_peaks+1))
        p1 = st.selectbox("Select Peak 1", peak_indices)
        p2 = st.selectbox("Select Peak 2", peak_indices)
        if p1!=p2:
            try:
                amp1, cen1, wid1 = popt[(p1-1)*3:(p1-1)*3+3]
                amp2, cen2, wid2 = popt[(p2-1)*3:(p2-1)*3+3]
                
                # Check for valid width values
                if wid1 <= 0 or wid2 <= 0:
                    st.error("Invalid peak widths detected. Check fitting results.")
                else:
                    FWHM1 = 2.3548*wid1
                    FWHM2 = 2.3548*wid2
                    Cf = abs(cen2-cen1)/(0.5*(FWHM1+FWHM2))
                    st.metric("Effective Seperation Coefficient Cf", f"{Cf:.3f}")
                    st.write("**Formula:** Cf = |Center2 - Center1| / (0.5*(FWHM1+FWHM2))")
                    st.write("If Cf > 1.5 → the peaks are generally considered baseline resolved (well separated).")
                    st.write("If Cf ≈ 1.0 → partial overlap (moderate separation).")
                    st.write("If Cf ≈ 1.0 → partial overlap (moderate separation).")




            except Exception as e:
                st.error(f"Error calculating coefficient: {str(e)}")
    else:
        st.info("Need at least 2 peaks for coefficient calculation")
else:
    st.info("Both datasets need to be fitted before computing Cf")
