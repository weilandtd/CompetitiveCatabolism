
from flask import Flask, render_template, request
app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    # Extract parameters from form
    param1 = float(request.form.get('param1', 0))
    param2 = float(request.form.get('param2', 0))

    # Run your model (placeholder function for now)
    result = param1 + param2  # Simple example calculation

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
