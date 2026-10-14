from ultralytics import YOLO
import gradio as gr
import PIL.Image as Image

model = YOLO("./weight/jic_yolov8n.pt")

def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img, 
        conf=conf_threshold, 
        iou=iou_threshold, 
        show_conf=True, 
        show_labels=True, 
        imgsz=640,)

    for r in results:
        img_array = r.plot()
        im = Image.fromarray(img_array[..., ::-1])

    return im

demo = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil"), 
        gr.Slider(minimum=0, maximum=1, value=0.1, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.3, label="IoU threshold")
    ],
    title="YOLOv8n Road Damage Detection",
    description=
        "Trained on ~20k images using RDD2022 dataset Japan, India, and China Motorbike subsets. " 
        "Detects four main road damage types: longitudinal crack, alligator crack, transverse crack, potholes. " 
        "Try the detector by uploading your own road images or use the examples below.",
    outputs=gr.Image(type="pil"),
    examples=[
        ["examples/jp1.jpg", 0.1, 0.3],
        ["examples/cz1.jpg", 0.1, 0.3]
    ],
)    

if __name__== "__main__":
    demo.launch(theme=gr.themes.Base())