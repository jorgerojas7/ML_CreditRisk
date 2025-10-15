from setuptools import find_packages, setup

setup(
    name='ml_creditrisk',
    packages=find_packages(),
    version='0.1.0',
    description='ML Credit Risk Analysis - Servicio de predicción de riesgo crediticio',
    author='Your Name',
    license='MIT',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'fastapi',
        'uvicorn',
        'pydantic',
        'matplotlib',
        'plotly',
        'streamlit',
        'requests',
        'python-dotenv',
        'jupyter',
        'pytest'
    ],
    python_requires='>=3.10',
)