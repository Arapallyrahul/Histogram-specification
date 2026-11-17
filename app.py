"""
Streamlit app: Contrast Enhancement Using Histogram Specification
Run:  streamlit run app.py
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="Histogram Specification", page_icon="🖼️", layout="wide")


def histogram_specification(source, reference):
    """Match source-image histogram to the reference-image histogram."""
    source_hist = np.bincount(source.ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference.ravel(), minlength=256).astype(np.float64)

    source_cdf = np.cumsum(source_hist) / source.size
    reference_cdf = np.cumsum(reference_hist) / reference.size

    mapping = np.interp(source_cdf, reference_cdf, np.arange(256))
    lookup_table = np.clip(np.round(mapping), 0, 255).astype(np.uint8)

    enhanced = cv2.LUT(source, lookup_table)
    return enhanced, source_hist, reference_hist, source_cdf, reference_cdf, lookup_table


def load_gray(uploaded_file):
    """Decode an uploaded image file into an 8-bit grayscale array."""
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def demo_images():
    """Generate a low-contrast source and a high-contrast reference."""
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(0, 1, 256), (256, 1))
    base = 0.5 * x + 0.5 * x.T
    low = (90 + 50 * base + rng.normal(0, 3, base.shape)).clip(0, 255).astype(np.uint8)
    ref = (255 * base ** 1.0 + rng.normal(0, 5, base.shape)).clip(0, 255).astype(np.uint8)
    return low, ref


st.title("🖼️ Contrast Enhancement using Histogram Specification")
st.caption("Digital Image Processing project – match the histogram of a source image to a reference image.")

with st.sidebar:
    st.header("Input")
    src_file = st.file_uploader("Low-contrast input image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    ref_file = st.file_uploader("Reference (target) image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    use_demo = st.checkbox("Use demo images", value=not (src_file and ref_file))
    st.markdown("---")
    st.markdown(
        "**Steps**\n"
        "1. Compute histograms\n"
        "2. Normalize to CDFs\n"
        "3. Map source CDF → reference CDF\n"
        "4. Apply lookup table to every pixel"
    )

if src_file and ref_file and not use_demo:
    source, reference = load_gray(src_file), load_gray(ref_file)
elif use_demo:
    source, reference = demo_images()
else:
    st.info("Upload both images in the sidebar, or tick **Use demo images**.")
    st.stop()

if source is None or reference is None:
    st.error("Could not read one of the images. Please upload a valid image file.")
    st.stop()

enhanced, s_hist, r_hist, s_cdf, r_cdf, lut = histogram_specification(source, reference)

# ---- Images ----
c1, c2, c3 = st.columns(3)
c1.image(source, caption="Input image", use_container_width=True, clamp=True)
c2.image(reference, caption="Reference image", use_container_width=True, clamp=True)
c3.image(enhanced, caption="Enhanced output", use_container_width=True, clamp=True)

# ---- Metrics ----
m1, m2, m3 = st.columns(3)
m1.metric("Input std-dev (contrast)", f"{source.std():.2f}")
m2.metric("Reference std-dev", f"{reference.std():.2f}")
m3.metric("Output std-dev", f"{enhanced.std():.2f}", delta=f"{enhanced.std() - source.std():+.2f}")

# ---- Graphs ----
e_hist = np.bincount(enhanced.ravel(), minlength=256)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
axes[0].plot(s_hist, label="Input")
axes[0].plot(r_hist, label="Reference")
axes[0].set_title("Histograms")
axes[0].legend()
axes[1].plot(s_cdf, label="Input CDF")
axes[1].plot(r_cdf, label="Reference CDF")
axes[1].set_title("CDF comparison")
axes[1].legend()
axes[2].plot(lut)
axes[2].set_title("Intensity mapping")
axes[2].set_xlabel("Input intensity")
axes[2].set_ylabel("Output intensity")
axes[3].plot(e_hist, label="Output", color="green")
axes[3].plot(r_hist, label="Reference", linestyle="--")
axes[3].set_title("Output vs Reference histogram")
axes[3].legend()
plt.tight_layout()
st.pyplot(fig)

# ---- Download ----
ok, buf = cv2.imencode(".png", enhanced)
if ok:
    st.download_button("⬇️ Download enhanced image", buf.tobytes(),
                       file_name="enhanced_output.png", mime="image/png")
