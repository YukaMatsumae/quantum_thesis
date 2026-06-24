# -*- coding: utf-8 -*-

"""
1D CNN PyTorch Demo

Task:
    Sine wave vs Square wave classification

Network:
    Conv1d
    -> ReLU
    -> MaxPool
    -> Conv1d
    -> Linear

"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt



# =========================
# 1. データ生成
# =========================


np.random.seed(0)
torch.manual_seed(0)


N = 2000
signal_length = 100


X = []
Y = []


for i in range(N):

    t = np.linspace(
        0,
        1,
        signal_length
    )


    # sine

    if i < N//2:


        signal = np.sin(
            2*np.pi*5*t
        )

        label = 0



    # square

    else:


        signal = np.sign(
            np.sin(2*np.pi*5*t)
        )

        label = 1



    # noise

    signal += 0.2*np.random.randn(
        signal_length
    )


    X.append(signal)
    Y.append(label)



X = np.array(X)
Y = np.array(Y)



# PyTorch形式

# Conv1d input:
# (batch, channel, length)


X = torch.tensor(
    X,
    dtype=torch.float32
)


X = X.unsqueeze(1)


Y = torch.tensor(
    Y,
    dtype=torch.float32
)



print("Input shape:", X.shape)





# =========================
# 2. train / test split
# =========================


train_size = int(
    N*0.8
)


X_train = X[:train_size]
Y_train = Y[:train_size]


X_test = X[train_size:]
Y_test = Y[train_size:]





# =========================
# 3. 1D CNNモデル
# =========================



class CNN1D(nn.Module):

    def __init__(self):

        super().__init__()



        self.conv = nn.Sequential(


            nn.Conv1d(

                in_channels=1,

                out_channels=32,

                kernel_size=5

            ),


            nn.ReLU(),


            nn.MaxPool1d(2),



            nn.Conv1d(

                32,

                64,

                kernel_size=5

            ),


            nn.ReLU(),


            nn.MaxPool1d(2)

        )



        self.fc = nn.Sequential(


            nn.Flatten(),


            nn.Linear(
                64*22,
                64
            ),


            nn.ReLU(),


            nn.Linear(
                64,
                1
            )

        )



    def forward(self,x):

        x = self.conv(x)

        x = self.fc(x)

        return x





model = CNN1D()


print(model)





# =========================
# 4. 学習設定
# =========================


criterion = nn.BCEWithLogitsLoss()


optimizer = optim.Adam(

    model.parameters(),

    lr=0.001

)





# =========================
# 5. Training
# =========================


epochs = 100


loss_history=[]



for epoch in range(epochs):


    model.train()


    optimizer.zero_grad()



    output = model(
        X_train
    ).squeeze()



    loss = criterion(
        output,
        Y_train
    )



    loss.backward()


    optimizer.step()



    loss_history.append(
        loss.item()
    )



    if (epoch+1)%5==0:


        print(
            f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}"
        )






# =========================
# 6. Accuracy評価
# =========================


model.eval()


with torch.no_grad():


    pred = torch.sigmoid(
        model(X_test)
    ).squeeze()



    predicted = (
        pred > 0.5
    ).float()



accuracy = (
    predicted == Y_test
).float().mean()



print(
    "\nTest Accuracy:",
    accuracy.item()
)





# =========================
# 7. Loss表示
# =========================


plt.plot(
    loss_history
)


plt.xlabel("Epoch")

plt.ylabel("Loss")


plt.title(
    "Training Loss"
)


plt.grid()

plt.show()





# =========================
# 8. 推論
# =========================


t = np.linspace(
    0,
    1,
    signal_length
)


# 元信号（理想波形）

true_signal = np.sign(
    np.sin(
        2*np.pi*5*t
    )
)



# CNN入力信号（ノイズ付き）

test_signal = true_signal + 0.2*np.random.randn(
    signal_length
)



# -------------------------
# 信号比較表示
# -------------------------


plt.figure(figsize=(10,4))


plt.plot(
    t,
    true_signal,
    label="Original signal"
)


plt.plot(
    t,
    test_signal,
    label="Input signal to CNN",
    alpha=0.7
)


plt.xlabel("Time")

plt.ylabel("Amplitude")


plt.title(
    "Original vs CNN Input Signal"
)


plt.legend()

plt.grid()


plt.show()





# -------------------------
# CNN推論
# -------------------------


test_tensor = torch.tensor(

    test_signal,

    dtype=torch.float32

).reshape(
    1,
    1,
    signal_length
)



model.eval()


with torch.no_grad():

    result = torch.sigmoid(
        model(test_tensor)
    )



print(
    "Prediction probability:",
    result.item()
)



if result.item() > 0.5:

    print("Result: Square wave")


else:

    print("Result: Sine wave")
