# StockPredictionWebApp

## Description

StockPredictionWebApp is an advanced web application designed to provide stock trend predictions using machine learning models.
## Features

- **Stock Data Retrieval:** Fetch historical stock price data from Yahoo Finance based on user input.
- **Interactive Charts:** Visualize stock price trends, moving averages, and predicted prices through interactive charts.
- **Streamlit Integration:** Offer a user-friendly and interactive web interface built with Streamlit.

## Installation

To set up the project locally, follow these steps:

```bash
git clone https://github.com/stelioszach03/StockPredictionWebApp.git
```
```bash
cd StockPredictionWebApp
```

Ensure that you have the necessary Python packages installed:

```bash
pip install numpy yfinance matplotlib pandas scikit-learn tensorflow streamlit
```

## Usage

To run the app locally:

```bash
streamlit run your_script_name.py
```

Navigate to `localhost:8501` in your web browser to view the app.

The app is also hosted and can be accessed directly at: [StockPredictionWebApp](http://164.92.67.220:8501/)

## How It Works

1. **Data Retrieval:** The app fetches historical stock data based on user input.
2. **Data Processing:** Implements data normalization and prepares it for the machine learning model.
3. **Visualization:** Displays interactive charts for price trends and comparisons between actual and predicted prices.

## Contributing

Contributions to the StockPredictionWebApp project are welcome! If you have suggestions to improve the application or have found a bug, please open an issue or submit a pull request.

## License

This project is open-source and available under the MIT License. See the [LICENSE.md](LICENSE) file for more details.
