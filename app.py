# ============================================================
# DeepGlobe AI - Land Cover Segmentation
# DeepLabV3+ + ResNet-50 + ImageNet Transfer Learning
# ============================================================

# Import Python utilities for handling files and HTML.
import io

# Import numerical processing.
import numpy as np

# Import image processing.
from PIL import Image

# Import PyTorch.
import torch
import torch.nn.functional as F

# Import Streamlit.
import streamlit as st

# Import segmentation models.
import segmentation_models_pytorch as smp


# ============================================================
# PAGE CONFIGURATION
# ============================================================

# Configure the Streamlit application page.
st.set_page_config(
    page_title="DeepGlobe AI | Land Cover Segmentation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

# Add the complete visual design to the application.
st.html("""
<style>

html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(30, 115, 255, 0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 30%,
            rgba(0, 210, 180, 0.08),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #020817 0%,
            #061525 50%,
            #03101c 100%
        );

    color: #f8fafc;
}

/* Remove Streamlit sidebar. */
[data-testid="stSidebar"] {
    display: none !important;
}

/* Remove sidebar collapse button. */
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Main application width. */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    padding: 65px 60px;
    margin-bottom: 35px;

    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(8, 37, 65, 0.98),
            rgba(7, 24, 44, 0.96)
        );

    border: 1px solid rgba(56, 189, 248, 0.30);

    box-shadow:
        0 25px 80px rgba(0, 0, 0, 0.35);

    position: relative;
    overflow: hidden;
}

.hero-kicker {
    color: #38bdf8;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 4px;
    margin-bottom: 18px;
}

.hero-title {
    color: #f8fafc;
    font-size: 56px;
    font-weight: 850;
    line-height: 1.05;
    letter-spacing: -2px;
    margin-bottom: 22px;
}

.hero-title span {
    color: #60a5fa;
}

.hero-description {
    color: #cbd5e1;
    font-size: 18px;
    line-height: 1.7;
    max-width: 850px;
    margin-bottom: 25px;
}

.hero-tags {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 4px;
}


/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.info-card {
    padding: 22px;
    min-height: 105px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(12, 37, 62, 0.95),
            rgba(7, 27, 47, 0.95)
        );

    border: 1px solid rgba(96, 165, 250, 0.18);
}

.info-label {
    color: #7187a3;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.info-value {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 750;
}


/* ============================================================
   SECTION HEADER
   ============================================================ */

.section-header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-top: 45px;
    margin-bottom: 10px;
}

.section-number {
    width: 45px;
    height: 45px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #38bdf8
        );

    color: white;
    font-weight: 800;
    font-size: 16px;

    box-shadow:
        0 8px 30px rgba(37, 99, 235, 0.35);
}

.section-title {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 800;
}

.section-description {
    color: #7f9abb;
    margin-left: 60px;
    margin-bottom: 25px;
    font-size: 14px;
}


/* ============================================================
   UPLOAD AREA
   ============================================================ */

.upload-card {
    padding: 25px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(8, 37, 63, 0.90),
            rgba(5, 25, 44, 0.90)
        );

    border: 1px solid rgba(56, 189, 248, 0.25);

    margin-bottom: 18px;
}

.upload-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 750;
    margin-bottom: 8px;
}

.upload-description {
    color: #7f9abb;
    font-size: 14px;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.70);
    border-radius: 18px;
    border: 1px dashed rgba(96, 165, 250, 0.50);
    padding: 12px;
}


/* ============================================================
   EMPTY STATE
   ============================================================ */

.empty-state {
    padding: 70px 30px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(8, 37, 63, 0.75),
            rgba(5, 25, 44, 0.75)
        );

    border: 1px solid rgba(56, 189, 248, 0.18);

    text-align: center;

    margin-top: 25px;
}

.empty-icon {
    font-size: 48px;
    margin-bottom: 18px;
}

.empty-title {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 10px;
}

.empty-text {
    color: #7187a3;
    font-size: 15px;
}


/* ============================================================
   RESULTS
   ============================================================ */

.result-header {
    padding: 25px 30px;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(14, 45, 75, 0.85),
            rgba(8, 29, 51, 0.85)
        );

    border: 1px solid rgba(56, 189, 248, 0.22);

    margin-top: 40px;
    margin-bottom: 25px;
}

.result-title {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 800;
}

.result-description {
    color: #7f9abb;
    font-size: 14px;
    margin-top: 5px;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {
    padding: 25px 15px;
    min-height: 125px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(20, 35, 55, 0.95),
            rgba(10, 25, 42, 0.95)
        );

    border: 1px solid rgba(148, 163, 184, 0.16);

    text-align: center;
}

.metric-name {
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    font-size: 29px;
    font-weight: 850;
}


/* ============================================================
   LEGEND
   ============================================================ */

.legend-container {
    padding: 25px;

    border-radius: 18px;

    background:
        rgba(8, 25, 43, 0.80);

    border: 1px solid rgba(56, 189, 248, 0.18);

    margin-top: 25px;
}

.legend-title {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 18px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 10px;

    color: #cbd5e1;

    font-size: 14px;

    margin-bottom: 10px;
}

.legend-dot {
    width: 14px;
    height: 14px;

    border-radius: 50%;

    display: inline-block;
}


/* ============================================================
   DOWNLOAD BUTTONS
   ============================================================ */

.stDownloadButton > button {
    width: 100%;

    border-radius: 12px;

    border: 1px solid rgba(96, 165, 250, 0.30);

    background:
        linear-gradient(
            135deg,
            rgba(25, 48, 78, 0.95),
            rgba(15, 35, 58, 0.95)
        );

    color: #f8fafc;

    font-weight: 700;

    padding: 13px;
}

.stDownloadButton > button:hover {
    border-color: #38bdf8;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    margin-top: 70px;

    padding: 40px 20px;

    text-align: center;

    border-top:
        1px solid rgba(148, 163, 184, 0.12);
}

.footer-highlight {
    color: #38bdf8;

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 3px;

    margin-bottom: 15px;
}

.footer-title {
    color: #cbd5e1;

    font-size: 16px;

    font-weight: 700;

    margin-bottom: 10px;
}

.footer-text {
    color: #607793;

    font-size: 13px;

    line-height: 1.8;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 800px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 40px 25px;
    }

    .hero-title {
        font-size: 38px;
    }

}

</style>
""")


# ============================================================
# MODEL SETTINGS
# ============================================================

# Define the trained model file.
MODEL_PATH = "best_deeplabv3plus.pth"

# Define the seven DeepGlobe classes.
CLASS_NAMES = [
    "Urban",
    "Agriculture",
    "Rangeland",
    "Forest",
    "Water",
    "Barren",
    "Unknown",
]

# Define the RGB colors used by the DeepGlobe dataset.
CLASS_COLORS = [
    (0, 255, 255),      # Urban
    (255, 255, 0),      # Agriculture
    (255, 0, 255),      # Rangeland
    (0, 255, 0),        # Forest
    (0, 0, 255),        # Water
    (255, 255, 255),    # Barren
    (0, 0, 0),          # Unknown
]

# Define colors used for the percentage cards.
DISPLAY_COLORS = [
    "#22d3ee",
    "#facc15",
    "#ff00ff",
    "#22c55e",
    "#2563eb",
    "#ffffff",
    "#64748b",
]


# ============================================================
# DEVICE
# ============================================================

# Automatically use GPU if CUDA is available.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

# Cache the model so it is loaded only once.
@st.cache_resource
def load_model():

    # Create the exact architecture used during training.
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=7,
    )

    # Load the trained checkpoint.
    try:

        # Load model weights using the current PyTorch format.
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

    except TypeError:

        # Fallback for older PyTorch versions.
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

    # Check whether the checkpoint contains a state_dict.
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        # Extract the state dictionary.
        state_dict = checkpoint["state_dict"]

    else:

        # Otherwise use the checkpoint directly.
        state_dict = checkpoint

    # Create a clean state dictionary.
    cleaned_state_dict = {}

    # Process every saved model parameter.
    for key, value in state_dict.items():

        # Remove "module." if the model was trained using DataParallel.
        new_key = key.replace(
            "module.",
            "",
            1,
        )

        # Store the cleaned parameter.
        cleaned_state_dict[new_key] = value

    # Load the trained parameters.
    model.load_state_dict(
        cleaned_state_dict
    )

    # Move the model to CPU or GPU.
    model = model.to(DEVICE)

    # Set the model to evaluation mode.
    model.eval()

    # Return the trained model.
    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

# Prepare an uploaded satellite image for model inference.
def preprocess_image(image):

    # Resize image to the same size used during training.
    image = image.resize(
        (512, 512),
        Image.Resampling.BILINEAR,
    )

    # Convert image to NumPy and scale pixels to 0-1.
    image_array = (
        np.array(image).astype(np.float32)
        / 255.0
    )

    # Convert HWC format into CHW format.
    image_array = np.transpose(
        image_array,
        (2, 0, 1),
    )

    # Convert NumPy array into a PyTorch tensor.
    tensor = torch.from_numpy(
        image_array
    ).float()

    # Define ImageNet mean values.
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    # Define ImageNet standard deviation values.
    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    # Apply ImageNet normalization.
    tensor = (
        tensor - mean
    ) / std

    # Add batch dimension.
    tensor = tensor.unsqueeze(0)

    # Move tensor to the selected device.
    tensor = tensor.to(DEVICE)

    # Return processed tensor.
    return tensor


# ============================================================
# CREATE COLORED MASK
# ============================================================

# Convert predicted class IDs into RGB colors.
def create_color_mask(class_mask):

    # Get the mask dimensions.
    height, width = class_mask.shape

    # Create an empty RGB mask.
    color_mask = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Apply each class color.
    for class_id, color in enumerate(
        CLASS_COLORS
    ):

        # Find pixels belonging to this class.
        pixels = class_mask == class_id

        # Assign the class color.
        color_mask[pixels] = color

    # Return the RGB mask.
    return Image.fromarray(
        color_mask
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

# Combine original satellite image and segmentation mask.
def create_overlay(
    original_image,
    color_mask,
):

    # Resize mask to original image size.
    color_mask = color_mask.resize(
        original_image.size,
        Image.Resampling.NEAREST,
    )

    # Blend original image and prediction.
    overlay = Image.blend(
        original_image,
        color_mask,
        0.45,
    )

    # Return overlay.
    return overlay


# ============================================================
# CALCULATE CLASS PERCENTAGES
# ============================================================

# Calculate the percentage occupied by every predicted class.
def calculate_percentages(class_mask):

    # Calculate total number of pixels.
    total_pixels = class_mask.size

    # Create result dictionary.
    percentages = {}

    # Calculate percentage for every class.
    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        # Count pixels of the current class.
        pixel_count = np.sum(
            class_mask == class_id
        )

        # Convert pixel count to percentage.
        percentage = (
            pixel_count
            / total_pixels
        ) * 100

        # Store percentage.
        percentages[class_name] = percentage

    # Return percentages.
    return percentages


# ============================================================
# HERO
# ============================================================

# Display the main hero section.
st.html("""
<div class="hero">

    <div class="hero-kicker">
        SATELLITE VISION &nbsp; | &nbsp; AI-POWERED ANALYSIS
    </div>

    <div class="hero-title">
        DeepGlobe <span>Land Cover</span> Segmentation
    </div>

    <div class="hero-description">
        Transform satellite imagery into meaningful land-cover
        insights using semantic segmentation powered by
        DeepLabV3+ and ResNet-50 transfer learning.
    </div>

    <div class="hero-tags">
        ANALYZE &nbsp;&nbsp; · &nbsp;&nbsp;
        VISUALIZE &nbsp;&nbsp; · &nbsp;&nbsp;
        UNDERSTAND
    </div>

</div>
""")


# ============================================================
# PROJECT INFORMATION
# ============================================================

# Create four information columns.
info_columns = st.columns(4)


with info_columns[0]:

    # Display model architecture.
    st.html("""
    <div class="info-card">
        <div class="info-label">
            Architecture
        </div>

        <div class="info-value">
            DeepLabV3+
        </div>
    </div>
    """)


with info_columns[1]:

    # Display encoder.
    st.html("""
    <div class="info-card">
        <div class="info-label">
            Encoder
        </div>

        <div class="info-value">
            ResNet-50
        </div>
    </div>
    """)


with info_columns[2]:

    # Display transfer learning information.
    st.html("""
    <div class="info-card">
        <div class="info-label">
            Transfer Learning
        </div>

        <div class="info-value">
            ImageNet
        </div>
    </div>
    """)


with info_columns[3]:

    # Display model input size.
    st.html("""
    <div class="info-card">
        <div class="info-label">
            Input Size
        </div>

        <div class="info-value">
            512 × 512
        </div>
    </div>
    """)


# ============================================================
# UPLOAD SECTION
# ============================================================

# Display upload section heading.
st.html("""
<div class="section-header">

    <div class="section-number">
        1
    </div>

    <div class="section-title">
        Upload a Satellite Image
    </div>

</div>

<div class="section-description">
    Upload an aerial or satellite image to begin
    AI-powered land-cover segmentation.
</div>
""")


# Display upload information.
st.html("""
<div class="upload-card">

    <div class="upload-title">
        🛰️ Satellite Image Input
    </div>

    <div class="upload-description">
        JPG, JPEG and PNG formats are supported.
        Upload an image to run the trained DeepLabV3+
        segmentation model.
    </div>

</div>
""")


# Create the image uploader.
uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    label_visibility="collapsed",
)


# ============================================================
# NO IMAGE
# ============================================================

# Show waiting screen if no image has been uploaded.
if uploaded_file is None:

    # Display the empty state.
    st.html("""
    <div class="empty-state">

        <div class="empty-icon">
            🛰️
        </div>

        <div class="empty-title">
            Ready for Satellite Analysis
        </div>

        <div class="empty-text">
            Upload a satellite image above to generate
            an AI-powered land-cover segmentation map
            and detailed class distribution.
        </div>

    </div>
    """)


# ============================================================
# IMAGE UPLOADED
# ============================================================

else:

    # Open the uploaded image.
    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ========================================================
    # ANALYSIS SECTION
    # ========================================================

    # Display analysis section.
    st.html("""
    <div class="section-header">

        <div class="section-number">
            2
        </div>

        <div class="section-title">
            Satellite Image Analysis
        </div>

    </div>

    <div class="section-description">
        The trained DeepLabV3+ model is analyzing
        the uploaded satellite image.
    </div>
    """)


    # Create two columns.
    image_columns = st.columns(
        [2.5, 1]
    )


    with image_columns[0]:

        # Display original satellite image.
        st.image(
            original_image,
            caption="Uploaded Satellite Image",
            use_container_width=True,
        )


    with image_columns[1]:

        # Display image metadata.
        runtime = (
            "CUDA GPU"
            if DEVICE.type == "cuda"
            else "CPU"
        )

        st.html(
            f"""
            <div class="info-card">

                <div class="info-label">
                    Image Size
                </div>

                <div class="info-value">
                    {original_image.width}
                    ×
                    {original_image.height}
                </div>

                <br>

                <div class="info-label">
                    File Format
                </div>

                <div class="info-value">
                    {uploaded_file.name.split(".")[-1].upper()}
                </div>

                <br>

                <div class="info-label">
                    Runtime
                </div>

                <div class="info-value">
                    {runtime}
                </div>

            </div>
            """
        )


    # ========================================================
    # RUN MODEL
    # ========================================================

    # Show progress while loading and predicting.
    with st.spinner(
        "Running DeepLabV3+ segmentation..."
    ):

        # Load trained model.
        model = load_model()

        # Preprocess uploaded image.
        input_tensor = preprocess_image(
            original_image
        )

        # Disable gradients for faster inference.
        with torch.no_grad():

            # Run model prediction.
            output = model(
                input_tensor
            )

            # Convert model output to probabilities.
            probabilities = F.softmax(
                output,
                dim=1,
            )

            # Select class with highest probability.
            prediction = torch.argmax(
                probabilities,
                dim=1,
            )

        # Move prediction from GPU to CPU.
        prediction = (
            prediction
            .squeeze(0)
            .cpu()
            .numpy()
        )


    # ========================================================
    # RESIZE PREDICTION
    # ========================================================

    # Convert class mask into a PIL image.
    prediction_image = Image.fromarray(
        prediction.astype(np.uint8)
    )

    # Resize mask to original image dimensions.
    prediction_image = prediction_image.resize(
        original_image.size,
        Image.Resampling.NEAREST,
    )

    # Convert mask back to NumPy.
    class_mask = np.array(
        prediction_image
    )


    # ========================================================
    # CREATE RESULTS
    # ========================================================

    # Create colored prediction mask.
    color_mask = create_color_mask(
        class_mask
    )

    # Create original + prediction overlay.
    overlay = create_overlay(
        original_image,
        color_mask,
    )

    # Calculate land-cover percentages.
    percentages = calculate_percentages(
        class_mask
    )


    # ========================================================
    # RESULT HEADER
    # ========================================================

    # Display result header.
    st.html("""
    <div class="result-header">

        <div class="result-title">
            ✨ Segmentation Results
        </div>

        <div class="result-description">
            Pixel-level land-cover prediction generated
            by DeepLabV3+.
        </div>

    </div>
    """)


    # ========================================================
    # RESULT IMAGES
    # ========================================================

    # Create three result columns.
    result_columns = st.columns(3)


    with result_columns[0]:

        # Display original image.
        st.image(
            original_image,
            caption="Original Satellite Image",
            use_container_width=True,
        )


    with result_columns[1]:

        # Display predicted segmentation.
        st.image(
            color_mask,
            caption="Predicted Land Cover",
            use_container_width=True,
        )


    with result_columns[2]:

        # Display segmentation overlay.
        st.image(
            overlay,
            caption="Segmentation Overlay",
            use_container_width=True,
        )


    # ========================================================
    # DISTRIBUTION SECTION
    # ========================================================

    # Display distribution heading.
    st.html("""
    <div class="section-header">

        <div class="section-number">
            3
        </div>

        <div class="section-title">
            Land Cover Distribution
        </div>

    </div>

    <div class="section-description">
        Percentage of image pixels assigned to each
        DeepGlobe land-cover category.
    </div>
    """)


    # ========================================================
    # FIRST FOUR METRICS
    # ========================================================

    # Create four metric columns.
    metric_columns = st.columns(4)


    # Display first four classes.
    for index in range(4):

        # Get class name.
        class_name = CLASS_NAMES[index]

        # Get percentage.
        percentage = percentages[
            class_name
        ]

        # Get display color.
        display_color = DISPLAY_COLORS[
            index
        ]

        with metric_columns[index]:

            # Display metric card.
            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-name">
                        {class_name}
                    </div>

                    <div
                        class="metric-value"
                        style="color:{display_color};"
                    >
                        {percentage:.2f}%
                    </div>

                </div>
                """
            )


    # ========================================================
    # REMAINING THREE METRICS
    # ========================================================

    # Create four columns for remaining classes.
    second_metric_columns = st.columns(4)


    # Display remaining three classes.
    for position, index in enumerate(
        range(4, 7)
    ):

        # Get class name.
        class_name = CLASS_NAMES[index]

        # Get percentage.
        percentage = percentages[
            class_name
        ]

        # Get display color.
        display_color = DISPLAY_COLORS[
            index
        ]

        with second_metric_columns[
            position
        ]:

            # Display metric card.
            st.html(
                f"""
                <div class="metric-card">

                    <div class="metric-name">
                        {class_name}
                    </div>

                    <div
                        class="metric-value"
                        style="color:{display_color};"
                    >
                        {percentage:.2f}%
                    </div>

                </div>
                """
            )


    # ========================================================
    # LEGEND
    # ========================================================

    # Display legend container.
    st.html("""
    <div class="legend-container">

        <div class="legend-title">
            🎨 Land Cover Legend
        </div>

    </div>
    """)


    # Create four legend columns.
    legend_columns = st.columns(4)


    # Display all seven classes.
    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        # Get RGB class color.
        rgb = CLASS_COLORS[index]

        # Convert RGB into CSS color.
        rgb_string = (
            f"rgb({rgb[0]}, "
            f"{rgb[1]}, "
            f"{rgb[2]})"
        )

        # Select column.
        column = legend_columns[
            index % 4
        ]

        with column:

            # Display legend item.
            st.html(
                f"""
                <div class="legend-item">

                    <span
                        class="legend-dot"
                        style="background:{rgb_string};"
                    ></span>

                    <span>
                        {class_name}
                    </span>

                </div>
                """
            )


    # ========================================================
    # DOWNLOAD SECTION
    # ========================================================

    # Display download heading.
    st.html("""
    <div class="section-header">

        <div class="section-number">
            4
        </div>

        <div class="section-title">
            Download Results
        </div>

    </div>

    <div class="section-description">
        Save the generated segmentation mask or
        visualization overlay.
    </div>
    """)


    # ========================================================
    # CREATE MASK DOWNLOAD
    # ========================================================

    # Create memory buffer for mask.
    mask_buffer = io.BytesIO()

    # Save mask into memory.
    color_mask.save(
        mask_buffer,
        format="PNG",
    )

    # Reset buffer position.
    mask_buffer.seek(0)


    # ========================================================
    # CREATE OVERLAY DOWNLOAD
    # ========================================================

    # Create memory buffer for overlay.
    overlay_buffer = io.BytesIO()

    # Save overlay into memory.
    overlay.save(
        overlay_buffer,
        format="PNG",
    )

    # Reset buffer position.
    overlay_buffer.seek(0)


    # ========================================================
    # DOWNLOAD BUTTONS
    # ========================================================

    # Create two download columns.
    download_columns = st.columns(2)


    with download_columns[0]:

        # Download segmentation mask.
        st.download_button(
            label="⬇️ Download Segmentation Mask",
            data=mask_buffer.getvalue(),
            file_name="deepglobe_segmentation_mask.png",
            mime="image/png",
            use_container_width=True,
        )


    with download_columns[1]:

        # Download segmentation overlay.
        st.download_button(
            label="⬇️ Download Overlay",
            data=overlay_buffer.getvalue(),
            file_name="deepglobe_segmentation_overlay.png",
            mime="image/png",
            use_container_width=True,
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    # Display model information heading.
    st.html("""
    <div class="section-header">

        <div class="section-number">
            ✓
        </div>

        <div class="section-title">
            Model Information
        </div>

    </div>
    """)


    # Create four information columns.
    model_columns = st.columns(4)


    with model_columns[0]:

        # Display architecture.
        st.html("""
        <div class="info-card">

            <div class="info-label">
                Architecture
            </div>

            <div class="info-value">
                DeepLabV3+
            </div>

        </div>
        """)


    with model_columns[1]:

        # Display encoder.
        st.html("""
        <div class="info-card">

            <div class="info-label">
                Encoder
            </div>

            <div class="info-value">
                ResNet-50
            </div>

        </div>
        """)


    with model_columns[2]:

        # Display output classes.
        st.html("""
        <div class="info-card">

            <div class="info-label">
                Output Classes
            </div>

            <div class="info-value">
                7 Land-Cover Classes
            </div>

        </div>
        """)


    with model_columns[3]:

        # Display inference device.
        st.html(
            f"""
            <div class="info-card">

                <div class="info-label">
                    Inference Device
                </div>

                <div class="info-value">
                    {
                        "CUDA GPU"
                        if DEVICE.type == "cuda"
                        else "CPU"
                    }
                </div>

            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

# Display the application footer.
st.html("""
<div class="footer">

    <div class="footer-highlight">
        SATELLITE INSIGHTS
        &nbsp; · &nbsp;
        AI VISION
        &nbsp; · &nbsp;
        SUSTAINABLE TECHNOLOGY
    </div>

    <div class="footer-title">
        DeepGlobe Land Cover Classification
    </div>

    <div class="footer-text">
        DeepLabV3+ · ResNet-50 · ImageNet Transfer Learning
        <br>
        Built with PyTorch & Streamlit
        <br><br>
        AI-powered semantic segmentation for satellite imagery.
    </div>

</div>
""")