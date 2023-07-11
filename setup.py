from setuptools import find_packages, setup

setup(
    name='netflix_customer_retention_package',
    packages=find_packages(),
    version='1.0.0',
    description='ML script to predict the customer retention based on Netflix userbase.',
    author='Shahriar Rahman',
    license='MIT License',
    author_email='shahriarrahman1101@gmail.com',
    url='https://github.com/shahriar-rahman/Netflix-Customer-Retention-using-GPR',
    python_requires='>=3.11, <4',
    install_requires=[
        'matplotlib~=3.7.1',
        'pandas~=2.0.0',
        'seaborn~=0.12.2',
        'sklearn~=0.0.post1',
        'scikit-learn~=1.2.2',
        'missingno~=0.5.2',
        'numpy~=1.24.2',
        'joblib~=1.2.0',
        'setuptools~=65.5.1',
    ],
)
