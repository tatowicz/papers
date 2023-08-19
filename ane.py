import torch
import torchvision

# Load a pre-trained version of MobileNetV2
torch_model = torchvision.models.mobilenet_v2(pretrained=True)

# Set the model in evaluation mode.
torch_model.eval()

# Trace the model with random data.
example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)


import coremltools as ct

# Coremltools doesn't work on python 3.11 so pyenv into a version that works (3.8 works)

# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)]
 )


# Save the converted model.
# Open he model in XCode to see the inputs and outputs
# Or open the model in netron
model.save("weights/newmodel.mlpackage")
