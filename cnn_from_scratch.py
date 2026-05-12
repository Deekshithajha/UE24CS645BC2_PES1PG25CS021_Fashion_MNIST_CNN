import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

print("CNN Assignment Started")

# =========================
# LOAD DATASET
# =========================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training images shape:", x_train.shape)
print("Testing images shape:", x_test.shape)

# =========================
# CONVOLUTION LAYER
# =========================

class ConvLayer:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                region = image[i:i+3, j:j+3]
                yield region, i, j

    def forward(self, input):
        self.last_input = input

        h, w = input.shape

        output = np.zeros((h - 2, w - 2, self.num_filters))

        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))

        return output

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * region

        self.filters -= learning_rate * d_L_d_filters

        return None

# =========================
# MAXPOOL LAYER
# =========================

class MaxPool:
    def iterate_regions(self, image):
        h, w, num_filters = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                region = image[(i * 2):(i * 2 + 2),
                               (j * 2):(j * 2 + 2)]

                yield region, i, j

    def forward(self, input):
        self.last_input = input

        h, w, num_filters = input.shape

        output = np.zeros((h // 2, w // 2, num_filters))

        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(region, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for region, i, j in self.iterate_regions(self.last_input):
            h, w, f = region.shape

            amax = np.amax(region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):

                        if region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2,
                                        j * 2 + j2,
                                        f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

# =========================
# SOFTMAX LAYER
# =========================

class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backward(self, d_L_d_out, learning_rate):
        for i, gradient in enumerate(d_L_d_out):

            if gradient == 0:
                continue

            t_exp = np.exp(self.last_totals)

            S = np.sum(t_exp)

            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            d_t_d_w = self.last_input
            d_t_d_inputs = self.weights

            d_L_d_t = gradient * d_out_d_t

            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]

            d_L_d_b = d_L_d_t

            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            self.weights -= learning_rate * d_L_d_w
            self.biases -= learning_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)

# =========================
# CNN MODEL
# =========================

conv = ConvLayer(8)
pool = MaxPool()
softmax = Softmax(13 * 13 * 8, 10)

# =========================
# FORWARD PASS
# =========================

def forward(image, label):
    out = conv.forward(image[:, :, 0])
    out = pool.forward(out)
    out = softmax.forward(out)

    loss = -np.log(out[label])

    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

# =========================
# TRAIN FUNCTION
# =========================

def train(image, label, lr=0.005):

    out, loss, acc = forward(image, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    gradient = softmax.backward(gradient, lr)
    gradient = pool.backward(gradient)
    conv.backward(gradient, lr)

    return loss, acc

# =========================
# TRAINING
# =========================

print("\n--- TRAINING STARTED ---\n")

epochs = 3

train_losses = []
train_accuracies = []

for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}")

    loss = 0
    num_correct = 0

    epoch_loss = 0
    epoch_acc = 0

    for i, (image, label) in enumerate(zip(x_train[:1000], y_train[:1000])):

        l, acc = train(image, label)

        loss += l
        num_correct += acc

        epoch_loss += l
        epoch_acc += acc

        if i % 100 == 99:
            print(
                f"[Step {i+1}] "
                f"Average Loss: {loss / 100:.3f} | "
                f"Accuracy: {num_correct}%"
            )

            loss = 0
            num_correct = 0

    avg_epoch_loss = epoch_loss / 1000
    avg_epoch_accuracy = (epoch_acc / 1000) * 100

    train_losses.append(avg_epoch_loss)
    train_accuracies.append(avg_epoch_accuracy)

# =========================
# TRAINING SUMMARY TABLE
# =========================

print("\n==============================")
print("TRAINING SUMMARY")
print("==============================")

print(f"{'Epoch':<10}{'Loss':<15}{'Accuracy (%)':<15}")

for i in range(epochs):
    print(f"{i+1:<10}{train_losses[i]:<15.4f}{train_accuracies[i]:<15.2f}")

# =========================
# TESTING
# =========================

print("\n--- TESTING ---\n")

loss = 0
num_correct = 0

for image, label in zip(x_test[:100], y_test[:100]):
    _, l, acc = forward(image, label)

    loss += l
    num_correct += acc

print("Test Loss:", round(loss / 100, 4))
print("Test Accuracy:", num_correct, "%")

print("\nCNN Assignment Completed")

# =========================
# PLOT TRAINING GRAPHS
# =========================

epochs_range = range(1, epochs + 1)

# Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_losses, marker='o')
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Accuracy Graph
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_accuracies, marker='o')
plt.title("Training Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()