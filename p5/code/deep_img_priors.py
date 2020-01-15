def __init__(self):
    super(EncDec, self).__init__()
    self.encoder = nn.Sequential(
    nn.Conv2d(1, 16, stride=2, padding=1, kernel_size=3),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    nn.Conv2d(16, 32, stride=2, padding=1, kernel_size=3),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.Conv2d(32, 64, stride=2, padding=1, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.Conv2d(64, 128, stride=2, padding=1, kernel_size=3),
    nn.BatchNorm2d(128),
    nn.ReLU(True)
    )
    self.decoder = nn.Sequential(
    nn.Upsample(align_corners=True, scale_factor=2, mode=’bilinear’),
    nn.Conv2d(128, 64, stride=1, padding=1, kernel_size=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Upsample(align_corners=True, scale_factor=2, mode=’bilinear’),
    nn.Conv2d(64, 32, stride=1, padding=1, kernel_size=3),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Upsample(align_corners=True, scale_factor=2, mode=’bilinear’),
    nn.Conv2d(32, 16, stride=1, padding=1, kernel_size=3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Upsample(align_corners=True, scale_factor=2, mode=’bilinear’),
    nn.Conv2d(16, 1, stride=1, padding=1, kernel_size=3),
    nn.ReLU(True)
    )

def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

def train(model, train_data, noisy_data, optimizer):
    # Squared error
    sse = F.mse_loss
    optimizer.zero_grad()
    output = model(train_data)
    a_loss = sse(output, noisy_data)
    loss = sse(output, noisy_data, reduction=’sum’)
    loss.backward()
    optimizer.step()
    print("Training Loss :%.4f Standard Error:%.4f"%(a_loss, loss))
    return loss, output


def test(model, random_data, clean_data, optimizer):
    with torch.no_grad():
    sse = F.mse_loss
    output = model(random_data)
    a_loss = sse(output, clean_data)
    test_loss = sse(output, clean_data, reduction=’sum’)
    print("Test Loss :%.4f Standard Error:%.4f"%(a_loss, test_loss))
    return test_loss

def train_test(model, random_data, noisy_data, clean_data, epochs,image):
    logger_train = Logger(’../logs/train/{}/{}’.format(image, 1))
    logger_test = Logger(’../logs/test/{}/{}’.format(image, 1))
    eta = 1e-02
    optimizer = optim.Adam(model.parameters(), lr=eta)
    train_loss = np.zeros((epochs))
    test_loss = np.zeros((epochs))


for epoch in range(1, epochs + 1):
    print("Epoch:{}".format(epoch))
    loss_train, loss_output = train(model, random_data, noisy_data, optimizer)
    loss_test = test(model, random_data, clean_data, optimizer)
    train_loss[epochs-1] = loss_train
    test_loss[epochs-1] = loss_test
    info_train = {’loss’: loss_train.item()}
    info_test = {’loss’: loss_test.item()}
    #Code reused from my CS682 assignments
    for tag, value in info_train.items():
    logger_train.scalar_summary(tag, value, epoch + 1)
    for tag, value in info_test.items():
    logger_test.scalar_summary(tag, value, epoch + 1)