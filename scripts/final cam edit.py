import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "best_model_dense.h5"
IMAGE_PATH = "example_brain_scan m.jpg"
OUTPUT_PATH = "final_report.jpg"

# CORRECT LAYER for EfficientNetB0 (based on your error log)
LAST_CONV_LAYER = "top_activation" 

CLASS_NAMES = ["brain_glioma", "brain_menin", "brain_tumor"]

# ===============================
# Load model
# ===============================
model = load_model(MODEL_PATH, compile=False)

# ===============================
# Image loader
# ===============================
def load_image(path):
    original = cv2.imread(path)
    if original is None:
        raise ValueError("Image not found")

    # Resize for display (High Res)
    original = cv2.resize(original, (512, 512))

    # Resize for Model Input (224, 224)
    img = cv2.resize(original, (224, 224))
    
    # Preprocessing (0-1 scale)
    img = img.astype(np.float32) / 255.0
    
    img = np.expand_dims(img, axis=0)
    return img, original

# ===============================
# STABLE GRAD-CAM
# ===============================
def gradcam(img_array, model, layer_name, class_idx):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    
    conv_outputs = conv_outputs[0]
    grads = grads[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)

    cam = np.maximum(cam.numpy(), 0)
    cam = cam / (np.max(cam) + 1e-8)

    return cam

# ===============================
# Final visualization (Soft Blend style)
# ===============================
def create_visual(original, heatmap, label):
    # 1. Resize heatmap to 512x512 to match original
    heatmap = cv2.resize(heatmap, (512, 512))
    
    # 2. Convert to uint8 (0-255)
    heatmap_uint8 = np.uint8(255 * heatmap)

    # 3. Apply Color Map to the WHOLE heatmap (not just the strong parts)
    # This creates the Blue -> Green -> Red gradient everywhere
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 4. SOFT BLEND (The key fix)
    # Mix original image (60%) with Heatmap (40%)
    # This lets the brain scan texture show through the color
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    # 5. Draw Yellow Circle
    # We use a threshold JUST to find the circle coordinates
    _, mask = cv2.threshold(heatmap_uint8, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest hot area
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        
        # Only draw if the circle is big enough (avoids noise)
        if r > 15:
            cv2.circle(overlay, (int(x), int(y)), int(r), (0, 255, 255), 3)

    # 6. Add Label Text
    text = f"Prediction: {label}"
    cv2.rectangle(overlay, (10, 10), (360, 48), (0, 0, 0), -1)
    cv2.putText(
        overlay, text, (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2
    )

    return overlay

# ===============================
# Run pipeline
# ===============================
def run():
    try:
        img_array, original = load_image(IMAGE_PATH)

        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        label = CLASS_NAMES[class_idx]
        print(f"Detected Class: {label}")

        heatmap = gradcam(
            img_array,
            model,
            LAST_CONV_LAYER,
            class_idx
        )

        final_img = create_visual(original, heatmap, label)
        cv2.imwrite(OUTPUT_PATH, final_img)

        print(f"Success! Output saved to -> {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()