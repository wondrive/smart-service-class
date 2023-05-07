import flwr as fl

from task import (
    Net,
    DEVICE,
    load_data,
    get_parameters,
    set_parameters,
    train,
    test,
)


# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, trainloader, epochs=1)
        return get_parameters(net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="0.0.0.0:9093",
    client=FlowerClient(),
    rest=True,
)
