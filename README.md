# Contrast Enhancement Using Histogram Specification

A Digital Image Processing (DIP) project that enhances the contrast of a low-contrast
grayscale image by matching its histogram to that of a reference image
(**histogram specification / histogram matching**). Includes a command-line script and an
interactive **Streamlit** web app.

## Features
- Upload a low-contrast input image and a reference (target) image
- Histogram specification implemented from scratch with NumPy (CDF mapping + lookup table)
- Side-by-side view of input, reference and enhanced output
- Plots: histograms, CDF comparison, intensity mapping, output vs reference histogram
- Contrast metric (standard deviation) before and after
- Download the enhanced image
- Built-in demo images if you have none handy

## How it works
1. **Histograms** – count pixels at each of the 256 intensity levels for both images.
2. **CDFs** – cumulatively sum and normalize each histogram to the range [0, 1].
3. **CDF mapping** – for every source intensity, find the reference intensity whose CDF value is closest (`np.interp`).
4. **Apply** – build a 256-entry lookup table and map every source pixel with `cv2.LUT`.

## Project structure
```
.
├── app.py                      # Streamlit web app
├── histogram_specification.py  # Core algorithm + command-line version
├── requirements.txt
└── README.md
```

## Installation
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Streamlit app
```bash
streamlit run app.py
```
Open the URL shown in the terminal (usually http://localhost:8501), upload the two images
in the sidebar, and view the results.

### Command line
Place `input_low_contrast.png` and `reference_target.png` in the project folder, then:
```bash
python histogram_specification.py
```
Outputs `enhanced_output.png` and `project_output_and_graphs.png`.

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to https://share.streamlit.io, click **New app**, choose your repo, branch and `app.py`.
3. Click **Deploy**.

## Tech stack
Python · OpenCV · NumPy · Matplotlib · Streamlit

## Author
Your Name – DIP Project
