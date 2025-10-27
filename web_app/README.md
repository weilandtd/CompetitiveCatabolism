# Competitive Catabolism Web Application

An interactive web application for simulating the multi nutrient model. 

## Quick Start

```bash
cd web_app

# Copy configuration template
cp .env.example .env

# Launch with auto-scaling
./launch.sh

# Or use the Python launcher directly
python3 start_server.py --mode production --autoscale
```

## Configuration

Edit `web_app/.env` to customize settings:
- `AUTO_SCALE=true` - Automatically scale workers based on CPU/load
- `WORKERS=4` - Fixed worker count (if AUTO_SCALE=false)
- `MIN_WORKERS=2` - Minimum workers for auto-scaling
- `MAX_WORKERS=16` - Maximum workers for auto-scaling
- `HOST=0.0.0.0` - Bind address
- `PORT=5000` - Server port

## Features

The web app provides five types of simulations:

### 1. Dynamic Response Simulation
- Perturb parameters at specified time points
- View time courses of glucose, fatty acids, ketones, and insulin
- Customizable parameter modifications

### 2. Insulin Clamp Studies
- Simulate hyperinsulinemic-euglycemic clamp experiments
- Optional metabolite co-infusions (lactate, fatty acids, ketones)
- Visualize glucose infusion rate (GIR) and metabolite concentrations

### 3. Tolerance Tests (GTT/ITT)
- Simulate Glucose Tolerance Tests and Insulin Tolerance Tests
- Compare control vs knockout responses

### 4. Obesity Simulations
- Explore parmer chanegs in the context of obeseity
- Compare the model predictions with NHANES human data
- Calculate HOMA-IR for insulin resistance assessment

### 5. Treatment Strategies
- Identify optimal therapeutic targets via sensitivity analysis
- Compare top 3 treatments to insulin therapy
- Visualize treatment efficacy

## Installation

1. Ensure you have Python 3.8+ installed

2. Install dependencies from the project root:
```bash
pip install -r requirements.txt
```

## Running the Application

### Simple Launch (Recommended):
```bash
# From project root
./launch.sh

# From web_app directory
cd web_app && ./launch.sh
```

### Development Mode:
```bash
./launch.sh development
# or
python3 start_server.py --mode development
```

### Production Mode with Auto-Scaling:
```bash
./launch.sh production
# or
python3 start_server.py --mode production --autoscale
```

### Advanced Options:
```bash
# Custom worker configuration
python3 start_server.py --mode production --workers 8 --no-autoscale

# Custom port
python3 start_server.py --mode production --port 8000

# Set worker limits
python3 start_server.py --mode production --min-workers 2 --max-workers 12
```

## File Structure

- **`app.py`** - Main Flask application
- **`wsgi.py`** - WSGI entry point for production servers
- **`start_server.py`** - Dynamic server with auto-scaling workers
- **`launch.sh`** - Simple launcher script
- **`config.py`** - Configuration management
- **`.env.example`** - Configuration template
- **`test_app.py`** - Application tests
- **`static/`** - CSS, JavaScript, images
- **`template/`** - HTML templates

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
