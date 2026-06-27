# app/pages/Model_Card.py
"""
Model card page.
"""
import streamlit as st

st.set_page_config(page_title="Model Card", page_icon="📋")
st.title("📋 Model Card")
st.markdown("""
### PCB Defect Detection Model

**Model**: YOLOv8n  
**Dataset**: DeepPCB (1,500 images, 6 defect classes)  
**Training**: 50 epochs, 480x480 input size  

### Performance
- **mAP@50**: 98.8%
- **mAP@50-95**: 74.3%
- **Precision**: 98.9%
- **Recall**: 96.2%
- **F1 Score**: 97.5%

### Deployment
- **PyTorch**: 57.8 FPS on M4 MPS
- **ONNX**: 54.8 FPS on CPU

### Limitations
- Best suited for 480x480 images
- May struggle with extreme lighting conditions
- Not yet tested on video streams
""")