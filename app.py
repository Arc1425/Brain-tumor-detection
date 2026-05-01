from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
from train import CNN

app = Flask(__name__)

# load model
model = CNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

classes = ["glioma", "meningioma", "pituitary", "no_tumor"]
descriptions = {
    "glioma": "Glioma is a type of tumor that occurs in the brain and spinal cord. It begins in the glial cells, which support nerve cells. Gliomas can grow aggressively and affect brain function depending on their location. Symptoms may include headaches, seizures, and cognitive difficulties. Early detection is important for treatment planning. Medical consultation is strongly recommended.",

    "meningioma": "Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain and spinal cord. It is usually slow-growing and often non-cancerous. However, depending on its size and location, it can press on brain structures. Common symptoms include headaches, vision problems, and memory issues. Most meningiomas are treatable if detected early. Regular monitoring is essential.",

    "pituitary": "Pituitary tumors develop in the pituitary gland, which regulates hormones in the body. These tumors can affect hormone production and lead to various health problems. Symptoms may include vision changes, fatigue, and hormonal imbalance. Most pituitary tumors are benign but require medical evaluation. Early diagnosis helps in effective management. Treatment may include medication or surgery.",

    "no_tumor": "No tumor detected in the brain MRI scan. This suggests that the brain structure appears normal with no visible abnormal growth. However, this does not replace professional medical diagnosis. If symptoms persist, further medical consultation is advised. Regular health checkups are important. Always rely on expert medical advice for confirmation."
}
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    probabilities = {classes[i]: round(probs[i].item() * 100, 2) for i in range(len(classes))}
    predicted_class = max(probabilities, key=probabilities.get)

    return predicted_class, probabilities

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).convert("RGB")

        result, probabilities = predict_image(img)
        description = descriptions[result]
        return render_template("index.html", result=result, probabilities=probabilities, description=description)
    return render_template("index.html")

if __name__ == "__main__":
    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)