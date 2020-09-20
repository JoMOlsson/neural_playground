# ----- Create data -----
r1 = 3
num_1 = 10000
r2 = 6
num_2 = 10000
train_data = np.zeros([num_1+num_2, 2])
train_labels = np.zeros([num_1+num_2, 1])
data = np.zeros([num_1, 2])
for i in range(0, num_1):
    r_diff = random.random()*r1
    theta_rand = random.random()*2*math.pi
    x = (r_diff) * math.cos(theta_rand)
    y = (r_diff) * math.sin(theta_rand)
    train_data[i, 0] = x
    train_data[i, 1] = y
    train_labels[i] = 0
    data[i, 0] = x
    data[i, 1] = y

data2 = np.zeros([num_1, 2])
inc = 0
for i in range(num_1, 2*num_2):
    r_diff = random.random()*(r2-r1) + r1 + 0.2
    theta_rand = random.random()*2*math.pi
    x = (r_diff) * math.cos(theta_rand)
    y = (r_diff) * math.sin(theta_rand)
    train_data[i, 0] = x
    train_data[i, 1] = y
    train_labels[i] = 1
    data2[inc, 0] = x
    data2[inc, 1] = y
    inc += 1

plt.scatter(data[:, 0], data[:, 1])
plt.axis('equal')
plt.scatter(data2[:, 0], data2[:, 1])
plt.show()
# -------------------