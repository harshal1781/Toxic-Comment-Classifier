
1. Create a new environment through Anaconda Prompt
- Enter 'conda env create -f toxic_test.yml'
- Enter 'python -m ipykernel install --user --name toxic --display-name toxic
- Enter 'conda activate toxic' (you can deactivate the env later)

2. Download Visual Studio Code (if you do not have it) to open the .py and .html files required for Flask

### How to run the Flask app: 
1. Open Anaconda Prompt
2. Change directory to 'Flask app for toxic comments' (cd .....)
3. Enter 'python toxic_app.py' to run the Python file
4. If you encounter a Module Not Found Error, enter 'pip install flask'
5. Copy the IP address where the Flask is running at (eg. 'http://127.0.0.1:5000/) into your browser
6. You should see the app in action

