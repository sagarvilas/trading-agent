from setuptools import setup, find_packages

setup(
    name="ai_trading_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "yfinance>=0.1.63",
        "schedule>=1.1.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered automated stock trading system",
)