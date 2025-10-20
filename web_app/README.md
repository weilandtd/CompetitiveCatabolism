# Competitive Catabolism Web Application

An interactive web application for simulating multi-nutrient metabolic dynamics based on competitive catabolism principles.

## Features

The web app provides five types of simulations:

### 1. Dynamic Response Simulation
- Simulate Type 1 Diabetes-like metabolic dynamics
- Perturb parameters at specified time points
- View time courses of glucose, fatty acids, ketones, and insulin
- Customizable parameter modifications

### 2. Insulin Clamp Studies
- Model hyperinsulinemic-euglycemic clamp experiments
- Test multiple insulin dose levels
- Optional metabolite co-infusions (lactate, fatty acids, ketones)
- Visualize glucose infusion rate (GIR) and metabolite concentrations

### 3. Tolerance Tests (GTT/ITT)
- Simulate Glucose Tolerance Tests and Insulin Tolerance Tests
- Model insulin receptor knockouts (AIRKO, MIRKO, LIRKO)
- Compare control vs knockout responses
- Adjustable bolus timing

### 4. Obesity Simulations
- Explore metabolic effects across adiposity spectrum
- Compare model predictions with NHANES human data
- Calculate HOMA-IR for insulin resistance assessment
- Test effects of parameter perturbations

### 5. Treatment Strategies
- Identify optimal therapeutic targets via sensitivity analysis
- Automatically rank treatments by HOMA-IR impact
- Compare top 3 perturbation-based treatments with insulin therapy
- Visualize treatment efficacy

## Installation

1. Ensure you have Python 3.8+ installed

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Development Mode

From the `web_app` directory:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Production Mode

Using Gunicorn (recommended for deployment):

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Project Structure

```
web_app/
├── app.py                  # Main Flask application with API endpoints
├── template/               # HTML templates
│   ├── index.html         # Landing page
│   ├── dynamic_response.html
│   ├── insulin_clamp.html
│   ├── tolerance_tests.html
│   ├── obesity.html
│   └── treatment.html
├── static/
│   └── style.css          # Shared stylesheet
└── README.md
```

## API Endpoints

- `GET /` - Landing page
- `GET /dynamic_response` - Dynamic response simulation page
- `GET /insulin_clamp` - Insulin clamp simulation page
- `GET /tolerance_tests` - Tolerance tests page
- `GET /obesity` - Obesity simulation page
- `GET /treatment` - Treatment strategies page

- `POST /api/run_dynamic` - Run dynamic response simulation
- `POST /api/run_clamp` - Run insulin clamp simulation
- `POST /api/run_tolerance_tests` - Run tolerance test simulation
- `POST /api/run_obesity` - Run obesity simulation
- `POST /api/run_treatment` - Run treatment analysis

## Usage Tips

### Parameter Modifications

All simulation pages allow you to:
- Modify primary parameters via dedicated controls
- Add additional parameter modifications using the "+ Add Parameter" button
- Download simulation results as CSV files

### Understanding Parameters

- **vE**: Energy expenditure (ATP production rate)
- **alpha**: Lipolysis activity (fat breakdown)
- **beta**: Fatty acid oxidation rate
- **gamma**: Glucose oxidation rate
- **Imax**: Maximum insulin secretion capacity
- **KI_lipo**: Insulin inhibition constant for lipolysis
- **KA_glut4**: Insulin activation constant for glucose uptake
- **VFK**: Ketogenesis activity
- **VLG**: Gluconeogenesis activity
- **VR**: Fatty acid reesterification activity

### Interpreting Results

- **HOMA-IR**: Homeostatic Model Assessment of Insulin Resistance (glucose × insulin)
- **GIR**: Glucose Infusion Rate (measure of insulin sensitivity in clamp studies)
- Higher HOMA-IR indicates greater insulin resistance
- Higher GIR during clamp indicates better insulin sensitivity

## Troubleshooting

### Import Errors

If you see import errors related to `multi_nutrient_model`, ensure:
1. You're running the app from the correct directory
2. The `multi_nutrient_model` directory exists at `../multi_nutrient_model` relative to `app.py`

### Matplotlib Backend Issues

If you encounter display issues, the app uses the 'Agg' backend (non-interactive) for server-side plotting. This is set in `app.py`.

### Simulation Errors

- Check that all input parameters are within reasonable ranges
- Fold changes should typically be between 0 and 5
- Time values should be positive
- Ensure comma-separated values are properly formatted

## Citation

If you use this web application in your research, please cite:

[Add appropriate citation here]

## License

[Add license information]

## Contact

For questions or issues, please contact [your contact information] or open an issue on GitHub.
