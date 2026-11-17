"""
Contrast Enhancement Using Histogram Specification
Requirements:
    pip install opencv-python numpy matplotlib

Usage:
    1. Put input_low_contrast.png and reference_target.png in this folder.
    2. Run: python histogram_specification.py
    3. The enhanced image and graphs are saved/displayed.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_specification(source, reference):
    """Match source-image histogram to the reference-image histogram."""
    # Step 1: Calculate histograms (256 intensity levels for 8-bit grayscale).
    source_hist = np.bincount(source.ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference.ravel(), minlength=256).astype(np.float64)

    # Step 2: Normalize cumulative histograms to obtain CDFs in [0, 1].
    source_cdf = np.cumsum(source_hist) / source.size
    reference_cdf = np.cumsum(reference_hist) / reference.size

    # Step 3: CDF mapping. For each source CDF value, find the
    # reference intensity having the closest corresponding CDF value.
    mapping = np.interp(source_cdf, reference_cdf, np.arange(256))
    lookup_table = np.clip(np.round(mapping), 0, 255).astype(np.uint8)

    # Step 4: Apply the mapping to every source pixel.
    enhanced = cv2.LUT(source, lookup_table)

    return enhanced, source_hist, reference_hist, source_cdf, reference_cdf, lookup_table


def main():
    source = cv2.imread("input_low_contrast.png", cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread("reference_target.png", cv2.IMREAD_GRAYSCALE)

    if source is None:
        raise FileNotFoundError("Cannot find input_low_contrast.png")
    if reference is None:
        raise FileNotFoundError("Cannot find reference_target.png")

    enhanced, source_hist, reference_hist, source_cdf, reference_cdf, lut = (
        histogram_specification(source, reference)
    )

    cv2.imwrite("enhanced_output.png", enhanced)
    print("Histogram specification completed.")
    print("Enhanced image saved as enhanced_output.png")

    # Display images and important graphs.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes[0, 0].imshow(source, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Input image")
    axes[0, 1].imshow(reference, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Reference image")
    axes[0, 2].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Enhanced output")

    axes[1, 0].plot(source_hist, label="Input")
    axes[1, 0].plot(reference_hist, label="Reference")
    axes[1, 0].set_title("Histograms")
    axes[1, 0].legend()

    axes[1, 1].plot(source_cdf, label="Input CDF")
    axes[1, 1].plot(reference_cdf, label="Reference CDF")
    axes[1, 1].set_title("CDF comparison")
    axes[1, 1].legend()

    axes[1, 2].plot(lut)
    axes[1, 2].set_title("Intensity mapping")
    axes[1, 2].set_xlabel("Input intensity")
    axes[1, 2].set_ylabel("Output intensity")

    for axis in axes.flat:
        if axis not in (axes[1, 0], axes[1, 1], axes[1, 2]):
            axis.axis("off")
    plt.tight_layout()
    plt.savefig("project_output_and_graphs.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
