# Flyc – Mobile Network Performance Analyzer

A modern Streamlit web app for analyzing and visualizing mobile network performance and coverage data. Designed for field engineers, analysts, and network professionals.

## Features
- **Interactive Data Upload**: Upload CSV/TXT files with measurement data.
- **Data Overview**: View key metrics, trends, and time series for selected bands and parameters.
- **Network Coverage Quality Map**: Visualize sensor locations, measurement points, and coverage polygons on an interactive map.
- **Performance Benchmarking**: Compare network performance by technology and operator.
- **Anomalies Detection**: Identify and visualize anomalies in performance and RF parameters with user-defined thresholds.
- **Customizable Filters**: Filter by date, operator, technology, modem, and more.
- **Modern UI**: Responsive, user-friendly interface with beautiful charts and maps.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/lugasia/flyc.git
   cd flyc
   ```
2. **Install dependencies:**
   (Recommended: use a virtual environment)
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally
```bash
streamlit run app.py
```
- The app will open in your browser at `http://localhost:8501`.

## Deploying to Streamlit Cloud
1. Push your code to a public GitHub repository (already done).
2. Go to [Streamlit Cloud](https://share.streamlit.io/), sign in, and click 'New app'.
3. Select your repository and branch, set `app.py` as the main file.
4. Click 'Deploy'.

## Example Usage
- Upload your measurement CSV file using the sidebar.
- Use the top filter bar to select date range, operator, technology, and modem(s).
- Switch between reports using the sidebar.
- In the Anomalies report, set thresholds and visualize anomalies graphically.

## Requirements
- Python 3.8+
- See `requirements.txt` for full list (Streamlit, pandas, plotly, scikit-learn, shapely, pillow, etc.)

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Contact
For questions or support, open an issue on GitHub or contact [Amir Lugasi](https://github.com/lugasia). 