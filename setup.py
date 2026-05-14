"""
Minimal setup.py to allow editable installs: pip install -e .
This ensures `from src.xxx import ...` works without PYTHONPATH hacks.
"""
from setuptools import setup, find_packages

setup(
    name="exam-stress-balancer",
    version="1.0.0",
    description="RL-powered exam stress balancer with full MLOps pipeline",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "mlflow>=2.13.0",
    ],
)
