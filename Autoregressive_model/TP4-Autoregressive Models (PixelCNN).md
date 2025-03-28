For each practical exercise (TP), please work in groups of two or three. Then, create a **private GitHub repository** and add me (my GitHub is **arthur-75**) to your project. Finally, share the link to your project (or TP) under  [Practical Exercises](https://docs.google.com/spreadsheets/d/1V-YKgHn71FnwjoFltDhWsPJS7uIuAh9lj6SP2DSCvlY/edit?usp=sharing) and make sure to choose your **team name** :-)

# **Autoregressive Models (PixelCNN)**

---

## **What is an Autoregressive Model in PixelCNN?**

Autoregressive models aim to learn the joint distribution of images by factorizing it into conditional distributions. For an image x, composed of pixels xi,j​:

![][image1]

where:

* x\<i,j​ refers to all pixels above and to the left of pixel xi,j.  
* PixelCNN strictly enforces this autoregressive dependency using **masked convolutions**.

---

## **The PixelCNN Architecture Explained**

PixelCNN generates images pixel-by-pixel, modeling the conditional probability distribution of each pixel based on previously generated pixels.

* **Masked Convolution Layers**  
   These ensure no pixel sees "future" pixels (pixels below or to the right), preserving the autoregressive property.

* **Residual Blocks**  
   Improve training stability and performance through skip connections.

---

## **Behind Masked Convolution**

### **Masked convolution filters enforce conditional dependence:**

* Type A mask: Used in the first convolutional layer (no current pixel allowed).  
* Type B mask: Used in subsequent layers (allows current pixel).

Given a convolutional kernel K, mask MMM, input image X:

![][image2]

Where:

* ⊙ represents element-wise multiplication.  
* Mask M ensures the convolution at pixel (i,j) does not include pixels at positions after (i,j).

---

## **Implementation** 

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt# Parameters
IMAGE_SIZE = 16
PIXEL_LEVELS = 4
N_FILTERS = 128
RESIDUAL_BLOCKS = 5
BATCH_SIZE = 128
EPOCHS = 150
DEVICE ="mps" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Data Preparation**

* Load FashionMNIST dataset.  
* Resize images to smaller dimensions (e.g., 16×16).  
* Quantize pixel values into fewer discrete levels (e.g., 4 levels) to simplify the probability distribution.

```py
# Data Preparation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x * 255).float() // (256 // PIXEL_LEVELS))
])
path= xxx
dataset = datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
```

**Masked Convolution Layer**

* Implement convolutional layers with masks:  
  * **Type A**: For the initial layer, exclude the pixel itself.  
  * **Type B**: For all subsequent layers, allow the current pixel.

```py
# Masked Convolution Layer
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, h, w = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
```

**Residual Blocks**

* Combine masked convolutions and non-linear activations with skip-connections:

```py
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(filters, filters // 2, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d('B', filters // 2, filters // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters // 2, filters, kernel_size=1)
        )

    def forward(self, x):
        return x + self.block(x)
```

**PixelCNN Model Structure**

* Stack masked convolution layers and residual blocks.  
* Output layer predicts probability distribution over pixel intensities using softmax activation.

```py
# PixelCNN Model
class PixelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            MaskedConv2d('A', 1, N_FILTERS, kernel_size=7, padding=3),
            *[ResidualBlock(N_FILTERS) for _ in range(RESIDUAL_BLOCKS)],
            nn.ReLU(),
            MaskedConv2d('B', N_FILTERS, N_FILTERS, kernel_size=1),
            nn.ReLU(),
            MaskedConv2d('B', N_FILTERS, N_FILTERS, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(N_FILTERS, PIXEL_LEVELS, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)

model = PixelCNN().to(DEVICE)
```

---

## **📉 5\. Training the PixelCNN**

### **Objective:**

Minimize the cross-entropy loss between predicted pixel distributions and actual pixel intensities.

**Cross-Entropy Loss**:\`\`

![][image3]

* yc​ is the true class label (pixel intensity category).  
* xc is the predicted probability for class c.

### **Training Loop (Conceptual):**

* Forward pass:  
  * Input: images (floats).  
  * Output: predicted probabilities for each pixel intensity.  
* Compute cross-entropy loss.  
* Backward pass and optimizer update (e.g., Adam optimizer).

```py
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, _ in tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        
        images = images.to(DEVICE).squeeze(1)
        optimizer.zero_grad()
        inputs = images.float().unsqueeze(1)     # [batch, 1, H, W]
        targets = images.long()                  # [batch, H, W]
        outputs = model(inputs)                  # [batch, PIXEL_LEVELS, H, W]       
        loss = criterion(outputs, targets)       # CrossEntropy expects float inputs and long targets 
        loss.backward()
        optimizer.step() 
        total_loss += loss.item()   
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

---

## **Sampling (Generating Images with PixelCNN)**

PixelCNN samples pixels sequentially from the learned distribution:

### **Sampling Steps:**

1. **Initialize**: Create an empty (zero-valued) image tensor.  
2. **Sequential Sampling**: Iterate through each pixel position (i,j):  
   * Compute conditional probability:   
     ![][image4]  
   * Sample pixel intensity from the probability distribution: 

   ![][image5]

   * Update pixel value in the generated image.

### **Role of Temperature T:**

* Controls randomness:  
  * Lower T → less randomness, sharper images.  
  * Higher T → more randomness, diverse images.

### **Formally:**

For each pixel, given logits z:

![][image6]

Then we sample from this distribution:

![][image7]

```py
def generate_images(model, num_images, temperature=1.0):
    model.eval()
    generated = torch.zeros(num_images, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    with torch.no_grad():
        for row in range(IMAGE_SIZE):
            for col in range(IMAGE_SIZE):
                logits = model(generated.float())[:, :, row, col] / temperature
                probs = torch.softmax(logits, dim=-1)
                generated[:, 0, row, col] = torch.multinomial(probs, 1).squeeze(-1)
    return generated.cpu().numpy() / PIXEL_LEVELS

# Generate sample images
sample_images = generate_images(model, num_images=10)
print("Generated images shape:", sample_images.shape)

```

---

## **Visualizing Generated Images**

* Generated images have pixel values between 0 and 1\.  
* Display images using visualization libraries (e.g., matplotlib) to inspect model quality.

```py
def plot_generated_images(images, n_cols=5):
    n_rows = (len(images) + n_cols - 1) // n_cols
    plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    for idx, img in enumerate(images):
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Generate sample images (assuming you've done this step already)
sample_images = generate_images(model, num_images=10, temperature=1.0)
# Plot generated images
plot_generated_images(sample_images)

```

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAANcAAAAkCAYAAAAae68CAAAJAUlEQVR4Xu2b56sVPRCH/QfsvTdExd7A3sWG7YNiQVHB3lBE7Ipib1gQRVGxooKifrArdrEXxAIqKnax95aXX2D2zc7JZrPn3r3NPBDO2UlOdjabSSaTnGzC4XDEQjYucDgcqYMzLocjJpxxORwx4YzL4YgJZ1xJULFiRS7KNDRv3pyLHAqNGzfmIh9FixblokCMxvX+/Xvx9etX8fv3b/H371/x8+dPea3y48cP3/WHDx9811mNChUqiOvXr3OxB57/169fsr3Qbt++fZPXxLt375TSQpZJKzp37izWrl3LxR5v376V7/jPnz8y4d1++fKFFxPfv3/3vuM5kdILtPfr16/Fq1ev5KcNeDaUp998/PjRy1u3bp1o0qTJ/4UZnz59Ejly5OBiLUbjqly5ssibN6/Inj27lwoUKODlHz161JeHlJlH9TBGjBghevfuzcUeW7duFWXKlElok5UrV3pl0H68PfmAFQfbt28XVapU4WKPCxcuiEqVKomcOXN6uhUuXFhUq1bNV27OnDk+/XPnzi0H4fQCfTRPnjyePmPHjuVFElD1L168uKhfv74vv2bNmmL8+PE+mcr69etFqVKluDgBo3ERI0eOlIqg4TmzZ88WRYoU8Y3OWREYANrABozk9PL4qE55R44c8cnjBve0eUcYQFC2S5cuPEuCUR/5K1as4FnpxtWrV0W7du2kXrVr1+bZPiZPniwmTpwoyy5btoxnS+gZTSD/9OnTXOzDyrgAdRa4DQTcI8xs/wJ169YVrVq14uJAaIbq1auXTw7Z06dPfbK4wWheqFAhLtYClwc6Xr58mWeJmzdvily5ciUMGLbARS5XrpysHzNevnz55Pd58+bxopHo06ePOHXqlNdHg4DRdOjQQbYFypmeo1ixYsb3vWDBAuO9gLVxYXZCZZ06dZLX8FXDKs9K4FmxJrHl3LlzCS8bBnfx4kWlVNoAHfbu3cvFWrjOxLhx40SzZs242JoZM2bITv348WOeJVavXi3vqQ7cUSB9g3Qn4OaCsHLg0KFDoWWQj3VbENbGhVmKlEIjhN04K7Fhw4aknpfa68SJE3Itum3bNl4kMlhwR+H+/fvWumOxrut4mG22bNnik0UBa9FatWpxsQ8EFfh9baHf6XQn9u3bJ86fPy9dY5TB2jgMlDN5GcgfPnw4F3tYGxcg5ZEwxf8rNGzYMPClmejXr5/XXpMmTeLZSQG3LAoTJkyw1n3KlCmybM+ePeU1rTN37drFStpDa0wb7ty5Ezkg9vDhQ9G2bVv5vUaNGoH3Klu2rPxctGiRLLNmzRpWIhGUGzBgABd7QNeg+4FIxjV06FBZGfzljMbixYvFwIEDI6Vp06bxarTgmRFViooa2EgJ6iK9QYMG8vPJkydepzKBtYPt/fFeUfbWrVti//793vrLNvSsY8iQIeLBgwdcHAjWYVHAezx58qT8PmrUKKkvv1+jRo287wULFrRuD5TDrB1E3759jXVFMi5SDMm0GEwPMJNiTaRL2FtCuBiuB1wf7N1gVOZ7dEHgeTF7RQWjI7XX4MGDebYVqoGi0wwbNkx2KFwjdB4G/dYGKov9sEGDBvlkyW4XBN27W7du2llhz549kSKRav07d+6U1wiVExiEMPASybRHENOnTzfmWxsX9kg2b97sBTZM+z2pwfHjx7ko3Ujmec+cOSN388+ePRv6kmy4cuWKtwdVr149GfmyAeVt9mRozYN06dIlT04ucdOmTZXS9uiem8L9SF27dvXlYVM4Slur9b98+VJewx0n1O0jihWQixhG+fLltfoTYWtxK+Nq3769t6l248aNVOksYaTEFUlt8Kzdu3fn4kAePXrke6nUXimJFC5dutSrB66TLuqmA+VtjuzQ3g9mRRV4BCl537rfqRvRfEZHAKZHjx4+WRCYyVu3bu2Toc7SpUvL73AT1dMX8+fPl/k26y1ABwKCIM8kiFDjgoIdO3b0yahhjh075pOnJialdWC0RVQqSoILYgN0wWxhw+fPnxN0h/sDmXq6xRY1godjRxiV4UHg2ia4QeumMGhWRICAQ3kLFy7kWaEE3Ruj/qxZs7hYBhxoDRUG2oL3QbWt+IBIe4+2UF1BIEhlyjca16pVq0TVqlW5WIwZM0ZWimMnQWCfBwtk/L5NmzaeHB2NXhbAPej4CsDJBXooRH+QbIBxIaoVJWHj0QboYrMJSzv7uvOCYS/KBNwook6dOvITLs7cuXM9eRDVq1e3uq9JPwQ3TPkm+vfvLzefbYnisej0IT11hxuiPgPKmlxqHBAw1RdoXLRoDoIU1XUkjCbqPgJ8YQRDCL4QpDOKBPYWTPdOa3CSPEwfRNhQ5tmzZzxLkj9/fpmPzpYSonQ+gIhomO7U3qZylJ+Ma2uqVwV7UXy2CeLNmzfaeknPa9eu+eS0rWAyFg7Kz5w5k4s94G6bIuc+49q9e7enHCU+O5FhqAkzEQ6GEpBxlwsy+NNg+fLlvoahjklkNOPCJnCQPnSER02qnw94PpIpxGuCd5owqFPpors4ccH1QsJsR9y9ezchHwb+4sULpSYzOBSMLQETCGDZDByIBHJ9SpQo4eW3aNHCF0VFVJKXR3/laz0dKPv8+XMu9kA+lk1BBM5cKQE31fnCOOQL4ArimsjoxgWgj249khmA7hs3buTiNAXrKF2/AAhK0NGkjMKOHTtC+2BYfmzGpe41kIxcCvynSFUMh0TVa7hWdK17GekB1jpRTw9kFDC62swKaQHcLLxbNWEvKqOBYNGSJUu42AP7jeqMqSMW49q0aZPPt713754vNH3gwAGfMfHjRVjH0bXpAdOSKMd4MiLQPcrB438ZitCaQH7YdkgsxgUOHz4sFUDURvffIIS2S5YsKTel4dfSKEbAV8ZicerUqcqv0hcEB/BHuszIwYMHk9oK+BdB8M0U4cSRLt3pEk5sxpVVadmypdxnyozgEC8da3LoQaADf48J4vbt29YnPJxxJcHo0aO5KNOQ0j8mZnUo6BaEzYxFOONyOGLCGZfDERPOuByOmHDG5XDEhDMuhyMmnHE5HDHhjMvhiAlnXA5HTDjjcjhi4j9bzFrEExAGVgAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAPoAAAAqCAYAAACa0kYSAAAJ00lEQVR4Xu2c56sUPRSH/Qfsvfcudj9YEMFeUVQQFXtBRRQLli+KIijqB3vBroiNK4iIHRv2XkAUwd4L9m5efrmceTNnMzPb7uzuveeBYScnmUw2mzPJOTmz+ZQgCLmefFwgCELuQxRdSAn58+fnorTkxYsXXBQa3759Uz9+/OBih1j6UBRdCB3bAIXMJo+X/fv3c1FcDBo0iItCpVGjRlzkIto+E0UXQqVfv356prIR7aANYu7cuUmpKxl1JAO/dvz+/Vt1796diyMQRRdCxW/Q+uXFyqJFi1zpp0+futJBTJkyRZUqVYqLU0JQvwTlA1F0ITSCBmRQfiLEWjfKv3r1iotTAsyQCRMmcLGLoO8nii6ERtBg5PlHjx7VsmnTpunPf//+ufIhq127tv6kg+RmXWa+KX/z5o2qWbOmWr16tZYfP37cdY0NquPZs2dq4cKFqnHjxmrMmDG8WNLxag9h6x8TUXQhFDAQv379ysUuzMH84MGDiMHNlZf4/Pmzb1lb2iYjRf/z509EHihYsKBzjvxjx46pGjVqWMsmm6B7fPnyxbeMKLoQCn6DkOCKvGHDBiM3WzZu3DjnnOfFkga1atXS8qZNm6pPnz458uvXr6tKlSoZJbOhGfP169fW+qJhwYIF6siRI1zs4FWvl9zEr4woupDjzJw5U1WsWJGLI+CKvmrVKiM3W1a4cGF93qBBAzVr1ix9DqfZqVOnjJKRg56nib179+o8HLt27dIyLOkLFCjASv5PkyZN1JAhQ7jYk2rVqnFRzHi136RMmTJq/PjxXKwRRRdyHAzSJ0+ecHEE5mAuWbKk6tq1q5GbnQ+7nc794Plmet26dREynuZ5gJbuyDPt4dmzZzvnxPnz511LfXDgwAHVokULl8wE/oZNmzZxscbWHs7jx489yyWk6EuWLOGiXMOZM2e4SIgTr8FnAsVBuQsXLjgypJctW6bPr1y5ogoVKuTKMw/TK419esh+/vzpyLASgFMP9jdh3g8PosWLF7vyTB49eqTreP/+vZ7t//79q+Vly5Z1ldu6dauz0uDgGl4vgQcb8Mr3W2GYeF3vKDrvODrgWbThVWFuonjx4lyUVsB5hAEQyzIyFSQ6VjDLITDEBHV+/PhRPyCg2H379g28z+nTp9Xbt2+d9IkTJ/Tn5s2b1a9fvxw5QF1btmxxyW7evOkE+2DGhjPOBA+BoEg2+AP8sC29IYt2qw/ttnnfXTM6KbcJ7BEuq1Chgiudm+HfPV3o0aOHmjFjhhoxYoTTRjiR0o0dO3YkvQ8x+ZA9bZLM+yDAJt768FCCvQynnUnnzp3159KlS/Un9yuMHj3alSZiaQfKrl27losjFZ3fDEsnfiOezs0UKVKEi9IC229gk6WaVq1a5Ui7YP9Onz5dL4ehWFhWr1mzhhdLiEOHDqnbt29zcUwMHjxYXbt2TZ8fPHjQFW03atQoxzQBtn7avn273j6MFtQBRyXHUXTaO4QtYtK8eXNXA2DLYJbPK8Cu2717NxenHNugsMlSDdpE9meygT0MZRk7dqz6/v07z04KyehTc+uOc/bsWefcdi/TLxENeJDY6nEU3cvGgezq1atOGk/ou3fvGiWyqV69uj5QHpv3derUidqBEC10j969e6uBAwfqe2zbto0XSzq2fkkWsBmxzKtcubK6dOmSK+/kyZP63h06dHDJd+7cqeX4xAEww3EZtqeGDh2q2rdvr+22Xr16qX379uk82KTwavO9atC/f3/9261fv94lh9cYfoHDhw/rNM7PnTun2+kF2sS950I28+fP15/oo27dujkOvkRAPbbx6nLGQXEQzD9p0iTVunVrLTO9lFSOYy4VYDuWLl1a3bt3z1o2Xsy9SNT77t07/WkLbEg2ft8DzhXkBx2mE8gED05i+fLlzjme5M+fP3fScAyakWW8TVhpcRmADA8BurZo0aL6wUGDitpnlsdDBwwbNkyVKFHCyQNYIpOtibJwZPmBMnjYCOEwfPhw6zhwKbrfS+6ErRJzQCLfNuMHhehBYaINLPCrJwi/pZCXRzSR+/lB2y03btxwyW0hmPhtuEKa+Cm6KYcH10wH2dC2PMiwIjDtSy9QFuaPEA4ITrL9ZlrR4cywZdrwK/fy5Uvf/GQQ6z3gY0iUWO4XK3PmzHGUke6DJZ3tnqaM50er6JMnT3al27ZtG3Ed0ljuw4vP80Cs4wUrAyEc0Ne230Yrute63oatXPny5fUnxQ4TdI5lI5wEfK+SGDBggOrZsycXu6AlJO4B+5zge5kAzg+8lWSC95MpfNKGXxts35m4fPmyysrKCjxscAcSZleYPrYXOrCPautbwtwOateunSOHzCw7depUVxpl/eql9MOHDx0ZTAHs3fv1J4HrTfNEyFlatmwZ8RuCfLQkDNroJzAD8Yg4qhjKSOemvQmnDvJo/9AEe8HA1jgCDjeyG1EP2Xz8mvv37+uZyIuOHTtykYacXbw+cOfOnRwbqFBePGQJ/D8ZbRGhLaYpVa5cOdf+qK2tJLNFkBHwv5jpNm3auNLm+Z49e5x0nz599Kc5OyMvaEsLZbidL+Qc2A62jY24QmB5RZixyRMLuLcW8Gs4QfnYhqB7YF8RkUkm8AD77XlTiKUftvyGDRtyUdKAGUIzNQ6sKkyQpjzTjq9SpYqqW7eujo3u1KmTI8c5ylKgBlZacLDiQKhmvXr19DVIY8urWLFi+hx1oU5Ann4c8BV06dJFlwN4JRPX4x1sYLbDi7Be4xSyQV/DGc5JiqIHgQigixcvOtswCBU0vfnYu0cEVbKggWmCgY1YZmxlAb4q8WpDrN9VcIN+zpQ+RDuxQkGoK15UQRpBMzClVqxYkRHfA22cN28eF8en6ICe6tGA2RRPffLGY/YlhQN+M0IiYLai2GQ4uDC7EM2aNXPOga0NYfxzSG4HL5ZkgoLgX2ZMKC7BhKfTEbTxw4cPXBy/opsOn3gw/7kyJzvQa/+aY2tD1apVuUiIA1vfphv8lVK0mbebp9MRrzbGreiJQMqH6KuNGze6wgDDgmxS/MBwFqWiDXkFr8GXTsC0NEGbeVwHdljSHa++TomiC3kLbMNRuGcmgH+YgcIgujOTgA/KK+xcFF3IcWxxAekMhTVnGmiz16vKouhCKGSS4qCtmRjN59fHouhCKEycOFG/UpoJ+ClMujJy5Ejfh5MouhAamaBACPXNhHZygtosii6ExsqVK/WfRKQj9evX12HGeO0Zf02NnRi+5Zau4B2NIGenKLoQKn5hykJ8+L16TYiiC6ETtMwUoifavhRFF1KCvNGWOPxVbD9E0YWUcOvWLS4SYsT8L8cgRNEFIQ8gii4IeQBRdEHIA4iiC0Ie4D9RBDQXHEMWzQAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,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>